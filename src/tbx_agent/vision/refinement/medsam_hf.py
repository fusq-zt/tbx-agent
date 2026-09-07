"""Verified, optional Hugging Face MedSAM box-prompt contour refiner.

This adapter accepts only D-FINE source-pixel boxes, runs them as SAM box
prompts in one image batch, and intersects every returned mask with the paired
PSPNet lung union.  It is deliberately not a diagnosis or routing component.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from ..anatomy import AnatomyEvidence, AnatomyQCStatus, decode_binary_mask, encode_binary_mask
from ..image_validator import ValidatedImage
from .base import (
    RefinementBackendError,
    RefinementBackendUnavailable,
    canonical_sha256,
    detector_box_digest,
)
from .models import (
    ContourRefinementEvidence,
    DetectionContourEvidence,
    DetectionRefinementStatus,
    RefinementRuntimeProbe,
)


@dataclass(frozen=True, slots=True)
class MedSAMRefinementPolicy:
    policy_id: str = "medsam-box-lung-constraint-qc-v1"
    probability_threshold: float = 0.5
    minimum_mask_pixels: int = 16
    maximum_mask_to_prompt_area_ratio: float = 4.0
    minimum_mask_prompt_overlap_fraction: float = 0.5
    max_prompts_per_run: int = 24
    max_batch_size: int = 4

    def __post_init__(self):
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("MedSAM probability threshold must lie in [0, 1]")
        if self.minimum_mask_pixels < 1:
            raise ValueError("MedSAM minimum mask size must be positive")
        if self.maximum_mask_to_prompt_area_ratio < 1.0:
            raise ValueError("MedSAM maximum mask/prompt ratio must be at least one")
        if not 0.0 <= self.minimum_mask_prompt_overlap_fraction <= 1.0:
            raise ValueError("MedSAM prompt-overlap threshold must lie in [0, 1]")
        if not 1 <= self.max_prompts_per_run <= 300:
            raise ValueError("MedSAM prompt limit must lie in [1, 300]")
        if not 1 <= self.max_batch_size <= self.max_prompts_per_run:
            raise ValueError("MedSAM batch size must lie within the prompt limit")

    def generation_parameters(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "probability_threshold": self.probability_threshold,
            "minimum_mask_pixels": self.minimum_mask_pixels,
            "maximum_mask_to_prompt_area_ratio": self.maximum_mask_to_prompt_area_ratio,
            "minimum_mask_prompt_overlap_fraction": (
                self.minimum_mask_prompt_overlap_fraction
            ),
            "constrain_to_lung_union": True,
            "max_prompts_per_run": self.max_prompts_per_run,
            "max_batch_size": self.max_batch_size,
        }


@dataclass(frozen=True, slots=True)
class HFMedSAMConfig:
    artifact_dir: str | Path
    expected_weight_sha256: str
    expected_config_sha256: str
    expected_preprocessor_sha256: str
    model_revision: str = "de8488bca37bb1d4fb190f612c516126d739ce3b"
    backend_id: str = "hf-wanglab-medsam-vit-base-box-refiner"
    model_id: str = "wanglab/medsam-vit-base"
    preprocessing_id: str = "hf-samprocessor-longest-edge-1024-rgb-v1"
    weight_filename: str = "pytorch_model.bin"
    config_filename: str = "config.json"
    preprocessor_filename: str = "preprocessor_config.json"
    device: str = "auto"

    def __post_init__(self):
        for label, value in (
            ("weight", self.expected_weight_sha256),
            ("config", self.expected_config_sha256),
            ("preprocessor", self.expected_preprocessor_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"expected MedSAM {label} SHA-256 is invalid")
        if re.fullmatch(r"[0-9a-f]{40}", self.model_revision) is None:
            raise ValueError("MedSAM model revision must be a full Git commit")
        for value in (
            self.weight_filename,
            self.config_filename,
            self.preprocessor_filename,
        ):
            if Path(value).name != value:
                raise ValueError("MedSAM artifact filenames must be plain filenames")


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validated_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise RefinementBackendError("D-FINE prompts must be finite xyxy boxes")
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise RefinementBackendError("D-FINE prompt lies outside the source image")
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def _anatomy_mask_digest(anatomy: AnatomyEvidence) -> str:
    return canonical_sha256(
        {
            "generation_key": anatomy.generation_key,
            "masks": [
                {
                    "structure": item.structure.value,
                    "sha256": item.payload.mask_sha256,
                    "width": item.payload.width,
                    "height": item.payload.height,
                }
                for item in anatomy.masks
            ],
        }
    )


def _lung_union(anatomy: AnatomyEvidence) -> list[list[bool]]:
    if anatomy.qc.status == AnatomyQCStatus.FAIL:
        raise RefinementBackendError("MedSAM refinement requires lung masks that passed QC")
    decoded = [decode_binary_mask(item.payload) for item in anatomy.masks]
    return [
        [left or right for left, right in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(decoded[0], decoded[1], strict=True)
    ]


def _box_lung_pixels(
    lung_union: list[list[bool]], box: tuple[int, int, int, int]
) -> int:
    x1, y1, x2, y2 = box
    return sum(sum(row[x1:x2]) for row in lung_union[y1:y2])


def _constrain_and_measure(
    raw_mask: Any,
    *,
    lung_union: list[list[bool]],
    box: tuple[int, int, int, int],
) -> tuple[list[list[bool]], int, int, float]:
    try:
        rows = [[bool(value) for value in row] for row in raw_mask]
    except (TypeError, ValueError) as exc:
        raise RefinementBackendError("MedSAM returned a non-2D mask") from exc
    height = len(lung_union)
    width = len(lung_union[0]) if height else 0
    if len(rows) != height or any(len(row) != width for row in rows):
        raise RefinementBackendError("MedSAM mask dimensions differ from the source image")
    raw_pixels = sum(sum(row) for row in rows)
    constrained = [
        [
            mask_value and lung_value
            for mask_value, lung_value in zip(mask_row, lung_row, strict=True)
        ]
        for mask_row, lung_row in zip(rows, lung_union, strict=True)
    ]
    constrained_pixels = sum(sum(row) for row in constrained)
    x1, y1, x2, y2 = box
    inside_prompt = sum(sum(row[x1:x2]) for row in constrained[y1:y2])
    prompt_fraction = inside_prompt / constrained_pixels if constrained_pixels else 0.0
    return constrained, raw_pixels, constrained_pixels, prompt_fraction


class HFMedSAMBoxRefinementBackend:
    """Lazy, hash-verified MedSAM backend with no routing authority."""

    def __init__(
        self,
        *,
        config: HFMedSAMConfig,
        policy: MedSAMRefinementPolicy | None = None,
    ):
        self.config = config
        self.policy = policy or MedSAMRefinementPolicy()
        self.backend_id = config.backend_id
        self._lock = RLock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._state_dict_digest: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._state_dict_digest is not None

    def _artifact_paths(self) -> tuple[Path, Path, Path]:
        root = Path(self.config.artifact_dir).expanduser().resolve()
        return (
            root / self.config.weight_filename,
            root / self.config.config_filename,
            root / self.config.preprocessor_filename,
        )

    def _verify_artifacts(self) -> None:
        paths = self._artifact_paths()
        expected = (
            self.config.expected_weight_sha256,
            self.config.expected_config_sha256,
            self.config.expected_preprocessor_sha256,
        )
        for path, digest in zip(paths, expected, strict=True):
            if not path.is_file():
                raise RefinementBackendUnavailable(
                    "pinned MedSAM artifact is missing; bootstrap medsam_refinement first"
                )
            if _file_sha256(path) != digest:
                raise RefinementBackendUnavailable(
                    "pinned MedSAM artifact failed its SHA-256 contract"
                )

    def probe_runtime(self, *, load: bool = False) -> RefinementRuntimeProbe:
        if self.loaded:
            return RefinementRuntimeProbe(
                backend_id=self.backend_id,
                loaded=True,
                available="yes",
                detail="runtime and exact MedSAM artifacts are loaded",
            )
        if load:
            try:
                self._load()
            except RefinementBackendUnavailable as exc:
                return RefinementRuntimeProbe(
                    backend_id=self.backend_id,
                    loaded=False,
                    available="no",
                    detail=str(exc),
                )
            return RefinementRuntimeProbe(
                backend_id=self.backend_id,
                loaded=True,
                available="yes",
                detail="runtime and exact MedSAM artifacts loaded successfully",
            )
        try:
            torch_spec = importlib.util.find_spec("torch")
            transformers_spec = importlib.util.find_spec("transformers")
        except (ImportError, AttributeError, ValueError):
            torch_spec = transformers_spec = None
        if torch_spec is None or transformers_spec is None:
            return RefinementRuntimeProbe(
                backend_id=self.backend_id,
                loaded=False,
                available="no",
                detail="optional torch/transformers MedSAM runtime is not installed",
            )
        if not all(path.is_file() for path in self._artifact_paths()):
            return RefinementRuntimeProbe(
                backend_id=self.backend_id,
                loaded=False,
                available="no",
                detail="one or more pinned MedSAM artifacts are missing",
            )
        return RefinementRuntimeProbe(
            backend_id=self.backend_id,
            loaded=False,
            available="unverified",
            detail="runtime and files are present but hashes/model load were not probed",
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._verify_artifacts()
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
                root = Path(self.config.artifact_dir).expanduser().resolve()
                processor = transformers.SamProcessor.from_pretrained(
                    root,
                    local_files_only=True,
                )
                model = transformers.SamModel.from_pretrained(
                    root,
                    local_files_only=True,
                    use_safetensors=False,
                )
                device_name = self.config.device
                if device_name == "auto":
                    device_name = "cuda" if torch.cuda.is_available() else "cpu"
                device = torch.device(device_name)
                model = model.to(device).eval()
                state_digest = _state_dict_sha256(model)
            except RefinementBackendUnavailable:
                raise
            except Exception as exc:
                raise RefinementBackendUnavailable(
                    "verified MedSAM artifacts could not be loaded by the isolated runtime"
                ) from exc
            self._torch = torch
            self._processor = processor
            self._model = model
            self._device = device
            self._state_dict_digest = state_digest

    def generation_key_for(
        self,
        *,
        image_sha256: str,
        boxes: list[tuple[float, float, float, float]],
        anatomy_generation_key: str,
    ) -> str:
        # The exact verified weight file fully determines the state dict.  Keep
        # key construction free of model loading so a missing optional runtime
        # can degrade only the refinement branch after the anatomy request has
        # been queued, rather than blocking PSPNet lung-field evidence.
        return canonical_sha256(
            {
                "schema": "tbx-contour-refinement-generation-v1",
                "image_sha256": image_sha256,
                "anatomy_generation_key": anatomy_generation_key,
                "detector_box_digest": detector_box_digest(boxes),
                "backend_id": self.backend_id,
                "model_id": self.config.model_id,
                "model_revision": self.config.model_revision,
                "weight_sha256": self.config.expected_weight_sha256,
                "config_sha256": self.config.expected_config_sha256,
                "preprocessor_sha256": self.config.expected_preprocessor_sha256,
                "preprocessing_id": self.config.preprocessing_id,
                "policy": self.policy.generation_parameters(),
            }
        )

    def _infer_masks(
        self,
        *,
        image: ValidatedImage,
        boxes: list[tuple[float, float, float, float]],
    ) -> list[Any]:
        assert self._torch is not None and self._model is not None
        assert self._processor is not None and self._device is not None
        inputs = self._processor(
            images=image.image.convert("RGB"),
            input_boxes=[[list(box) for box in boxes]],
            return_tensors="pt",
        )
        moved = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self._model(**moved, multimask_output=False)
        try:
            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                binarize=False,
            )[0]
            probabilities = masks.sigmoid()
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise RefinementBackendError("MedSAM output failed post-processing") from exc
        if probabilities.ndim != 4 or probabilities.shape[0] != len(boxes):
            raise RefinementBackendError("MedSAM output does not cover every box prompt")
        return [
            (probabilities[index, 0] >= self.policy.probability_threshold)
            .detach()
            .cpu()
            .numpy()
            .tolist()
            for index in range(len(boxes))
        ]

    def refine(
        self,
        *,
        case_id: str,
        image: ValidatedImage,
        boxes: list[tuple[float, float, float, float]],
        anatomy: AnatomyEvidence,
    ) -> ContourRefinementEvidence:
        started = time.perf_counter()
        self._load()
        assert self._state_dict_digest is not None
        if (
            anatomy.case_id != case_id
            or anatomy.image_sha256 != image.sha256
            or anatomy.image_width != image.width
            or anatomy.image_height != image.height
        ):
            raise RefinementBackendError("MedSAM inputs violate the immutable case identity")
        lung_union = _lung_union(anatomy)
        integer_boxes = [
            _validated_box(box, width=image.width, height=image.height) for box in boxes
        ]
        eligible_indexes = [
            index
            for index, box in enumerate(integer_boxes)
            if index < self.policy.max_prompts_per_run
            and _box_lung_pixels(lung_union, box) > 0
        ]
        eligible_masks: dict[int, Any] = {}
        for offset in range(0, len(eligible_indexes), self.policy.max_batch_size):
            batch_indexes = eligible_indexes[offset : offset + self.policy.max_batch_size]
            inferred = self._infer_masks(
                image=image,
                boxes=[boxes[index] for index in batch_indexes],
            )
            eligible_masks.update(dict(zip(batch_indexes, inferred, strict=True)))

        items: list[DetectionContourEvidence] = []
        for index, (source_box, integer_box) in enumerate(
            zip(boxes, integer_boxes, strict=True)
        ):
            if index >= self.policy.max_prompts_per_run:
                items.append(
                    DetectionContourEvidence(
                        detection_index=index,
                        bbox_xyxy=source_box,
                        status=DetectionRefinementStatus.CAPACITY_ABSTAINED,
                        raw_mask_pixels=0,
                        lung_constrained_pixels=0,
                        note="refinement_capacity_limit",
                    )
                )
                continue
            if index not in eligible_masks:
                items.append(
                    DetectionContourEvidence(
                        detection_index=index,
                        bbox_xyxy=source_box,
                        status=DetectionRefinementStatus.OUTSIDE_LUNGS,
                        raw_mask_pixels=0,
                        lung_constrained_pixels=0,
                        note="detection_box_has_no_lung_overlap",
                    )
                )
                continue
            constrained, raw_pixels, constrained_pixels, prompt_fraction = (
                _constrain_and_measure(
                    eligible_masks[index],
                    lung_union=lung_union,
                    box=integer_box,
                )
            )
            x1, y1, x2, y2 = integer_box
            prompt_area = (x2 - x1) * (y2 - y1)
            passes_qc = (
                constrained_pixels >= self.policy.minimum_mask_pixels
                and constrained_pixels
                <= prompt_area * self.policy.maximum_mask_to_prompt_area_ratio
                and prompt_fraction >= self.policy.minimum_mask_prompt_overlap_fraction
            )
            if not passes_qc:
                items.append(
                    DetectionContourEvidence(
                        detection_index=index,
                        bbox_xyxy=source_box,
                        status=DetectionRefinementStatus.MASK_QC_FAILED,
                        raw_mask_pixels=raw_pixels,
                        lung_constrained_pixels=0,
                        mask_prompt_overlap_fraction=prompt_fraction,
                        note="refinement_mask_qc_failed",
                    )
                )
                continue
            items.append(
                DetectionContourEvidence(
                    detection_index=index,
                    bbox_xyxy=source_box,
                    status=DetectionRefinementStatus.REFINED,
                    mask=encode_binary_mask(constrained),
                    raw_mask_pixels=raw_pixels,
                    lung_constrained_pixels=constrained_pixels,
                    mask_prompt_overlap_fraction=prompt_fraction,
                    note="visualization_only_nonvalidated_contour",
                )
            )

        generation_key = self.generation_key_for(
            image_sha256=image.sha256,
            boxes=boxes,
            anatomy_generation_key=anatomy.generation_key,
        )
        return ContourRefinementEvidence(
            case_id=case_id,
            image_sha256=image.sha256,
            image_width=image.width,
            image_height=image.height,
            backend_id=self.backend_id,
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            model_weight_sha256=self.config.expected_weight_sha256,
            model_state_dict_sha256=self._state_dict_digest,
            model_config_sha256=self.config.expected_config_sha256,
            preprocessor_config_sha256=self.config.expected_preprocessor_sha256,
            preprocessing_id=self.config.preprocessing_id,
            policy_id=self.policy.policy_id,
            generation_key=generation_key,
            anatomy_generation_key=anatomy.generation_key,
            anatomy_mask_digest=_anatomy_mask_digest(anatomy),
            detector_box_digest=detector_box_digest(boxes),
            items=items,
            max_prompts_per_run=self.policy.max_prompts_per_run,
            max_batch_size=self.policy.max_batch_size,
            capacity_abstained_count=max(
                0,
                len(boxes) - self.policy.max_prompts_per_run,
            ),
            runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )
