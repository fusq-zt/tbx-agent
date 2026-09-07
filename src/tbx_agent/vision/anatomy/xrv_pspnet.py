"""Optional TorchXRayVision ChestX-Det PSPNet adapter.

The dependency and its managed weights are loaded only at first inference.  A
missing optional runtime produces an explicit availability error instead of a
mock mask or a silent fallback.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image

from ..image_validator import ValidatedImage
from .base import AnatomyBackendError, AnatomyBackendUnavailable, build_generation_key
from .models import AnatomyEvidence, AnatomyMask, AnatomyRuntimeProbe, LungSide
from .qc import AnatomyQCPolicy, evaluate_lung_masks
from .rle import encode_binary_mask


@dataclass(frozen=True, slots=True)
class XRVPSPNetConfig:
    backend_id: str = "torchxrayvision-chestx-det-pspnet"
    model_id: str = "torchxrayvision:chestx_det.PSPNet"
    preprocessing_id: str = "xrv-normalize-centercrop-resize-512-v1"
    input_size: int = 512
    probability_threshold: float = 0.5
    left_lung_channel: int = 4
    right_lung_channel: int = 5
    device: str = "auto"
    cache_dir: str | Path | None = None
    weight_filename: str = "pspnet_chestxray_best_model_4.pth"
    expected_weight_file_sha256: str | None = None

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError("input size must be positive")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability threshold must lie in [0, 1]")
        if self.left_lung_channel < 0 or self.right_lung_channel < 0:
            raise ValueError("lung channels must be non-negative")
        if self.left_lung_channel == self.right_lung_channel:
            raise ValueError("left and right lungs require distinct channels")
        if Path(self.weight_filename).name != self.weight_filename:
            raise ValueError("weight filename must be a plain filename")
        if (
            self.expected_weight_file_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.expected_weight_file_sha256) is None
        ):
            raise ValueError("expected weight-file SHA-256 must be a lowercase digest")


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_weight_file(path: Path, *, expected_sha256: str | None) -> str:
    if not path.is_file():
        raise AnatomyBackendUnavailable(
            "TorchXRayVision PSPNet did not expose the expected checkpoint file for verification."
        )
    observed = _file_sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise AnatomyBackendUnavailable(
            "TorchXRayVision PSPNet checkpoint SHA-256 does not match the pinned artifact."
        )
    return observed


def _center_crop_geometry(width: int, height: int) -> tuple[int, int, int]:
    side = min(width, height)
    return (width - side) // 2, (height - side) // 2, side


def _restore_mask_to_source(
    mask: object,
    *,
    source_width: int,
    source_height: int,
) -> list[list[bool]]:
    """Invert the deterministic center-crop/resize transform."""

    import numpy as np

    raw = np.asarray(mask, dtype=np.uint8)
    if raw.ndim != 2:
        raise AnatomyBackendError("PSPNet lung output must be a two-dimensional mask")
    crop_x, crop_y, side = _center_crop_geometry(source_width, source_height)
    resized = Image.fromarray(raw * 255, mode="L").resize((side, side), Image.Resampling.NEAREST)
    restored = np.zeros((source_height, source_width), dtype=bool)
    restored[crop_y : crop_y + side, crop_x : crop_x + side] = (
        np.asarray(resized, dtype=np.uint8) > 0
    )
    return restored.tolist()


class TorchXRayVisionPSPNetBackend:
    """Lazy, routing-neutral adapter for ChestX-Det's PSPNet lung masks."""

    def __init__(
        self,
        *,
        config: XRVPSPNetConfig | None = None,
        qc_policy: AnatomyQCPolicy | None = None,
    ):
        self.config = config or XRVPSPNetConfig()
        self.qc_policy = qc_policy or AnatomyQCPolicy()
        self.backend_id = self.config.backend_id
        self._lock = RLock()
        self._model: Any | None = None
        self._torch: Any | None = None
        self._xrv: Any | None = None
        self._device: Any | None = None
        self._weight_sha256: str | None = None
        self._state_dict_sha256: str | None = None
        self._weight_file_path: Path | None = None

    @property
    def loaded(self) -> bool:
        return (
            self._model is not None
            and self._weight_sha256 is not None
            and self._state_dict_sha256 is not None
        )

    def probe_runtime(self, *, load: bool = False) -> AnatomyRuntimeProbe:
        if self.loaded:
            return AnatomyRuntimeProbe(
                backend_id=self.backend_id,
                loaded=True,
                available="yes",
                detail="runtime and model weights are loaded",
            )
        if load:
            try:
                self._load()
            except AnatomyBackendUnavailable as exc:
                return AnatomyRuntimeProbe(
                    backend_id=self.backend_id,
                    loaded=False,
                    available="no",
                    detail=str(exc),
                )
            return AnatomyRuntimeProbe(
                backend_id=self.backend_id,
                loaded=True,
                available="yes",
                detail="runtime and model weights loaded successfully",
            )
        try:
            torch_spec = importlib.util.find_spec("torch")
            xrv_spec = importlib.util.find_spec("torchxrayvision")
        except (ImportError, AttributeError, ValueError):
            torch_spec = None
            xrv_spec = None
        if torch_spec is None or xrv_spec is None:
            return AnatomyRuntimeProbe(
                backend_id=self.backend_id,
                loaded=False,
                available="no",
                detail="optional torch/torchxrayvision runtime is not installed",
            )
        return AnatomyRuntimeProbe(
            backend_id=self.backend_id,
            loaded=False,
            available="unverified",
            detail="optional runtime is present; model weights have not been loaded or verified",
        )

    def _generation_parameters(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "model_state_dict_sha256": self._state_dict_sha256,
            "input_size": self.config.input_size,
            "probability_threshold": self.config.probability_threshold,
            "left_lung_channel": self.config.left_lung_channel,
            "right_lung_channel": self.config.right_lung_channel,
            "qc": self.qc_policy.generation_parameters(),
        }

    def generation_key_for(self, image_sha256: str) -> str:
        self._load()
        assert self._weight_sha256 is not None
        return build_generation_key(
            image_sha256=image_sha256,
            model_weight_sha256=self._weight_sha256,
            preprocessing_id=self.config.preprocessing_id,
            policy_id=self.qc_policy.policy_id,
            backend_id=self.backend_id,
            parameters=self._generation_parameters(),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                torch = importlib.import_module("torch")
                xrv = importlib.import_module("torchxrayvision")
            except ImportError as exc:
                raise AnatomyBackendUnavailable(
                    "TorchXRayVision PSPNet is optional; install the anatomy/vision runtime "
                    "and allow its pinned weight bootstrap before enabling this backend."
                ) from exc
            device_name = self.config.device
            if device_name == "auto":
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                cache_dir = self._resolve_cache_dir(xrv)
                weight_path = cache_dir / self.config.weight_filename
                before_sha256 = None
                if (
                    self.config.expected_weight_file_sha256 is not None
                    and not weight_path.is_file()
                ):
                    raise AnatomyBackendUnavailable(
                        "pinned TorchXRayVision PSPNet checkpoint is missing; run the artifact "
                        "bootstrap before enabling anatomy inference"
                    )
                if weight_path.exists():
                    # Verify before torch.load can deserialize the checkpoint.
                    before_sha256 = _verify_weight_file(
                        weight_path,
                        expected_sha256=self.config.expected_weight_file_sha256,
                    )
                constructor = xrv.baseline_models.chestx_det.PSPNet
                parameters = inspect.signature(constructor).parameters
                if "cache_dir" in parameters:
                    model = constructor(cache_dir=str(cache_dir))
                elif self.config.cache_dir is not None:
                    default_cache = self._default_xrv_cache_dir(xrv)
                    if default_cache != cache_dir:
                        raise AnatomyBackendUnavailable(
                            "installed TorchXRayVision PSPNet does not accept cache_dir; "
                            "refusing to bypass the verified bootstrap cache"
                        )
                    model = constructor()
                else:
                    model = constructor()
                after_sha256 = _verify_weight_file(
                    weight_path,
                    expected_sha256=self.config.expected_weight_file_sha256,
                )
                if before_sha256 is not None and before_sha256 != after_sha256:
                    raise AnatomyBackendUnavailable(
                        "TorchXRayVision PSPNet checkpoint changed while the model was loading."
                    )
                model = model.to(torch.device(device_name)).eval()
                state_dict_sha256 = _state_dict_sha256(model)
            except AnatomyBackendUnavailable:
                raise
            except Exception as exc:
                raise AnatomyBackendUnavailable(
                    "TorchXRayVision PSPNet or its managed weights could not be loaded."
                ) from exc
            self._torch = torch
            self._xrv = xrv
            self._device = torch.device(device_name)
            self._model = model
            self._weight_sha256 = after_sha256
            self._state_dict_sha256 = state_dict_sha256
            self._weight_file_path = weight_path

    def _resolve_cache_dir(self, xrv: Any) -> Path:
        if self.config.cache_dir is not None:
            return Path(self.config.cache_dir).expanduser().resolve()
        return self._default_xrv_cache_dir(xrv)

    @staticmethod
    def _default_xrv_cache_dir(xrv: Any) -> Path:
        get_cache_dir = getattr(getattr(xrv, "utils", None), "get_cache_dir", None)
        if not callable(get_cache_dir):
            raise AnatomyBackendUnavailable(
                "TorchXRayVision cache directory cannot be resolved for checkpoint verification."
            )
        return Path(get_cache_dir()).expanduser().resolve()

    @staticmethod
    def _extract_output(output: Any) -> Any:
        if isinstance(output, dict):
            for key in ("out", "pred", "logits"):
                if key in output:
                    return output[key]
            raise AnatomyBackendError("PSPNet returned no recognized segmentation tensor")
        if isinstance(output, (tuple, list)):
            if not output:
                raise AnatomyBackendError("PSPNet returned an empty output")
            return output[0]
        return output

    def _preprocess(self, image: ValidatedImage) -> Any:
        import numpy as np

        assert self._torch is not None and self._xrv is not None
        grayscale = np.asarray(image.image.convert("L"), dtype=np.float32)
        normalized = self._xrv.datasets.normalize(grayscale, 255)
        sample = normalized[None, ...]
        sample = self._xrv.datasets.XRayCenterCrop()(sample)
        sample = self._xrv.datasets.XRayResizer(self.config.input_size)(sample)
        return self._torch.from_numpy(sample).float().unsqueeze(0).to(self._device)

    def infer(self, *, case_id: str, image: ValidatedImage) -> AnatomyEvidence:
        started = time.perf_counter()
        self._load()
        assert self._torch is not None and self._model is not None
        assert self._weight_sha256 is not None
        assert self._state_dict_sha256 is not None
        with self._lock, self._torch.inference_mode():
            output = self._extract_output(self._model(self._preprocess(image)))
            if getattr(output, "ndim", None) != 4 or output.shape[0] != 1:
                raise AnatomyBackendError("PSPNet must return a [1,C,H,W] tensor")
            required_channel = max(
                self.config.left_lung_channel,
                self.config.right_lung_channel,
            )
            if output.shape[1] <= required_channel:
                raise AnatomyBackendError("PSPNet output does not contain configured lung channels")
            values = output.float()
            if float(values.min()) < 0.0 or float(values.max()) > 1.0:
                values = values.sigmoid()
            left_small = (
                values[0, self.config.left_lung_channel]
                >= self.config.probability_threshold
            ).detach().cpu().numpy()
            right_small = (
                values[0, self.config.right_lung_channel]
                >= self.config.probability_threshold
            ).detach().cpu().numpy()

        # Resolve rare channel overlap using per-pixel probabilities, preserving
        # exactly one side before restoring source coordinates.
        overlap = left_small & right_small
        if overlap.any():
            left_probability = values[0, self.config.left_lung_channel].detach().cpu().numpy()
            right_probability = values[0, self.config.right_lung_channel].detach().cpu().numpy()
            left_wins = left_probability >= right_probability
            left_small = left_small & (~overlap | left_wins)
            right_small = right_small & (~overlap | ~left_wins)

        left = _restore_mask_to_source(
            left_small,
            source_width=image.width,
            source_height=image.height,
        )
        right = _restore_mask_to_source(
            right_small,
            source_width=image.width,
            source_height=image.height,
        )
        _, _, crop_side = _center_crop_geometry(image.width, image.height)
        crop_fraction = 1.0 - (crop_side * crop_side) / (image.width * image.height)
        qc = evaluate_lung_masks(
            left,
            right,
            policy=self.qc_policy,
            source_width=image.width,
            source_height=image.height,
            preprocessing_crop_fraction=crop_fraction,
        )
        generation_key = self.generation_key_for(image.sha256)
        return AnatomyEvidence(
            run_id=str(uuid.uuid4()),
            case_id=case_id,
            image_sha256=image.sha256,
            image_width=image.width,
            image_height=image.height,
            backend_id=self.backend_id,
            model_id=self.config.model_id,
            model_weight_sha256=self._weight_sha256,
            model_state_dict_sha256=self._state_dict_sha256,
            preprocessing_id=self.config.preprocessing_id,
            policy_id=self.qc_policy.policy_id,
            generation_key=generation_key,
            masks=[
                AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
                AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right)),
            ],
            qc=qc,
            runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )
