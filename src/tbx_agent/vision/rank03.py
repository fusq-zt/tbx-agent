from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..config import resolve_model_path
from ..schemas import ClassifierClass, DetectionEvidence, VisionEvidence
from .base import VisionBackendError
from .image_validator import ValidatedImage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise VisionBackendError(f"rank03 {label} 缺失：{path}")
    # Local interactive startup can bind already-reviewed external checkpoints
    # without re-reading hundreds of megabytes on every process start.  Model
    # construction and strict state-dict loading still fail closed below.
    if os.getenv("TBX_AGENT_LOCAL_FAST_START", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    actual = _sha256(path)
    if actual != expected_sha256:
        raise VisionBackendError(
            f"rank03 {label} SHA256 不匹配；期望 {expected_sha256}，实际 {actual}"
        )


def _native_classifier_argmax(
    probabilities: dict[str, float],
    order: list[str],
) -> tuple[ClassifierClass | None, bool]:
    expected_order = [item.value for item in ClassifierClass]
    if order != expected_order or set(probabilities) != set(expected_order):
        raise VisionBackendError("ConvNeXt 类别顺序必须严格为 healthy, sick_non_tb, tb。")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities.values()
    ):
        raise VisionBackendError("ConvNeXt 类别概率必须是 [0, 1] 内的有限值。")
    maximum = max(probabilities.values())
    winners = [ClassifierClass(name) for name in order if probabilities[name] == maximum]
    # Match the declared ``argmax([healthy, sick_non_tb, tb])`` contract:
    # ties use the stable class order instead of creating a fourth review
    # route.  The tie flag remains available as internal uncertainty metadata.
    return winners[0], len(winners) != 1


def _policy_decision_fields(
    policy: dict[str, Any],
    *,
    probabilities: dict[str, float],
    predicted_class: ClassifierClass | None,
    argmax_tied: bool,
    detections: list[DetectionEvidence],
) -> dict[str, Any]:
    classifier_rule = str(policy.get("classifier_rule", ""))
    detector_role = policy.get("detector_role")
    if classifier_rule == "native_three_class_argmax":
        if detector_role != "advisory_localization_only":
            raise VisionBackendError(
                "native argmax policy requires detector_role=advisory_localization_only"
            )
        return {
            "classifier_decision_rule": "native_three_class_argmax",
            "predicted_class": predicted_class,
            "classifier_argmax_tied": argmax_tied,
            "classifier_threshold": None,
            "classifier_flagged": predicted_class == ClassifierClass.TB,
            "detector_decision_role": "advisory_localization_only",
            "detector_threshold": None,
            "detector_flagged": None,
        }

    if classifier_rule == "p_tb_gte_threshold" and detector_role == "advisory_localization_only":
        if "detector_rule" in policy or "detector_threshold" in policy:
            raise VisionBackendError(
                "advisory detector policy cannot contain a detector decision rule or threshold"
            )
        try:
            raw_threshold = policy["classifier_threshold"]
            if isinstance(raw_threshold, bool):
                raise TypeError
            classifier_threshold = float(raw_threshold)
        except (KeyError, TypeError, ValueError) as exc:
            raise VisionBackendError("p_tb decision policy threshold is invalid") from exc
        if not math.isfinite(classifier_threshold) or not 0.0 <= classifier_threshold <= 1.0:
            raise VisionBackendError("p_tb decision policy threshold must be in [0, 1]")
        return {
            "classifier_decision_rule": "p_tb_gte_threshold",
            # Keep native argmax only as descriptive model evidence. Fusion below
            # is controlled exclusively by classifier_flagged.
            "predicted_class": predicted_class,
            "classifier_argmax_tied": argmax_tied,
            "classifier_threshold": classifier_threshold,
            "classifier_flagged": probabilities["tb"] >= classifier_threshold,
            "detector_decision_role": "advisory_localization_only",
            "detector_threshold": None,
            "detector_flagged": None,
        }

    detector_rule = str(policy.get("detector_rule", ""))
    legacy_detector = (
        detector_role is None or detector_role == "legacy_vote"
    ) and detector_rule in {"", "max_category_agnostic_score_gte_threshold"}
    if classifier_rule != "p_tb_gte_threshold" or not legacy_detector:
        raise VisionBackendError("unsupported or mixed classifier/detector decision policy")
    try:
        classifier_threshold = float(policy["classifier_threshold"])
        detector_threshold = float(policy["detector_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionBackendError("legacy decision policy thresholds are invalid") from exc
    if not 0.0 <= classifier_threshold <= 1.0 or not 0.0 <= detector_threshold <= 1.0:
        raise VisionBackendError("legacy decision policy thresholds must be in [0, 1]")
    max_detector_score = max((item.score for item in detections), default=0.0)
    return {
        "classifier_decision_rule": "legacy_p_tb_threshold",
        "predicted_class": None,
        "classifier_argmax_tied": False,
        "classifier_threshold": classifier_threshold,
        "classifier_flagged": probabilities["tb"] >= classifier_threshold,
        "detector_decision_role": "legacy_vote",
        "detector_threshold": detector_threshold,
        "detector_flagged": max_detector_score >= detector_threshold,
    }


class Rank03Backend:
    """Fail-closed adapter for one hash-bound, independently installed rank03 bundle.

    The active policy uses native three-class argmax. D-FINE detections are advisory
    localization evidence and cannot override the classifier route.
    """

    backend_id = "tbx11k-rank03-official-a-v1"
    runtime_contract = "rank03-frozen-runtime-v1"
    synthetic = False

    def __init__(self, policy: dict[str, Any], runtime_config: dict[str, Any]):
        self.policy = policy
        self.config = runtime_config
        if runtime_config.get("template") is True:
            raise VisionBackendError(
                "rank03 推理权重尚未安装：请下载发布的模型包，运行 "
                "scripts/install_vision_bundle.py，并设置 TBX_AGENT_RANK03_RUNTIME_CONFIG。"
            )
        classifier = runtime_config["classifier"]
        detector = runtime_config["detector"]
        self.classifier_checkpoint = resolve_model_path(
            classifier["checkpoint_path"], "TBX_AGENT_CLASSIFIER_CHECKPOINT"
        )
        self.classifier_config = resolve_model_path(
            classifier["config_path"], "TBX_AGENT_CLASSIFIER_CONFIG"
        )
        self.detector_checkpoint = resolve_model_path(
            detector["checkpoint_path"], "TBX_AGENT_DETECTOR_CHECKPOINT"
        )
        self.detector_resolved_config = resolve_model_path(
            detector["resolved_config_path"], "TBX_AGENT_DETECTOR_CONFIG"
        )
        self.dfine_root = resolve_model_path(detector["source_root"], "TBX_AGENT_DFINE_ROOT")
        self._classifier = None
        self._detector = None
        self._postprocessor = None
        self._device = None
        self._lock = threading.Semaphore(1)
        self._verify_frozen_assets()

    def probe_runtime(self) -> dict[str, Any]:
        """Load both frozen models so readiness cannot pass on paths and hashes alone.

        This deliberately does not manufacture a screening result.  It proves that
        PyTorch, timm, the D-FINE source tree, both state dicts, and the selected
        compute device can construct the exact runtime used by ``infer``.
        """

        with self._lock:
            self._load_classifier()
            self._load_detector()
            loaded = (
                self._classifier is not None
                and self._detector is not None
                and self._postprocessor is not None
                and self._device is not None
            )
            if not loaded:
                raise VisionBackendError("rank03 运行时探针未能加载全部冻结模型。")
            return {
                "classifier_loaded": True,
                "detector_loaded": True,
                "device": str(self._device),
                "image_inference_performed": False,
            }

    def _verify_frozen_assets(self) -> None:
        classifier = self.config["classifier"]
        detector = self.config["detector"]
        _verify_asset(
            self.classifier_checkpoint,
            classifier["checkpoint_sha256"],
            "ConvNeXt-Tiny checkpoint",
        )
        _verify_asset(
            self.classifier_config,
            classifier["config_sha256"],
            "ConvNeXt-Tiny config",
        )
        _verify_asset(
            self.detector_checkpoint,
            detector["checkpoint_sha256"],
            "D-FINE-L checkpoint",
        )
        _verify_asset(
            self.detector_resolved_config,
            detector["resolved_config_sha256"],
            "D-FINE-L resolved config",
        )
        if not self.dfine_root.is_dir():
            raise VisionBackendError(f"D-FINE source root 缺失：{self.dfine_root}")

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise VisionBackendError("rank03 后端需要可用的 PyTorch。") from exc
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch

    def _load_classifier(self):
        if self._classifier is not None:
            return self._classifier
        torch = self._torch()
        try:
            import timm
        except ImportError as exc:
            raise VisionBackendError("rank03 分类器需要 timm；请安装 vision extra。") from exc

        cfg = self.config["classifier"]
        on_disk_cfg = json.loads(self.classifier_config.read_text(encoding="utf-8"))
        contract = (cfg["architecture"], int(cfg["input_size"]), bool(cfg["amp"]))
        observed = (
            on_disk_cfg.get("model"),
            int(on_disk_cfg.get("input_size", -1)),
            bool(on_disk_cfg.get("amp", False)),
        )
        if observed != contract:
            raise VisionBackendError(f"ConvNeXt config 漂移：{observed!r} != {contract!r}")
        state = torch.load(self.classifier_checkpoint, map_location="cpu", weights_only=True)
        if state.get("config", {}).get("model") != cfg["architecture"]:
            raise VisionBackendError("ConvNeXt checkpoint 内嵌架构与冻结配置不一致。")
        model = timm.create_model(
            cfg["architecture"], pretrained=False, num_classes=3, drop_rate=0.0
        )
        incompatible = model.load_state_dict(state["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise VisionBackendError(f"ConvNeXt state dict 不兼容：{incompatible}")
        self._classifier = model.to(self._device).eval()
        del state
        return self._classifier

    def _load_detector(self):
        if self._detector is not None and self._postprocessor is not None:
            return self._detector, self._postprocessor
        torch = self._torch()
        try:
            sys.path.insert(0, str(self.dfine_root))
            from src.core import YAMLConfig  # type: ignore[import-not-found]
        except Exception as exc:
            raise VisionBackendError(
                "D-FINE 运行依赖不可用；请安装 vision extra（含 faster-coco-eval、tensorboard）。"
            ) from exc

        resolved = json.loads(self.detector_resolved_config.read_text(encoding="utf-8"))
        if resolved.get("num_classes") != 1:
            raise VisionBackendError("D-FINE num_classes 漂移。")
        if resolved.get("remap_mscoco_category") is not False:
            raise VisionBackendError("D-FINE category remapping 漂移。")
        if resolved.get("eval_spatial_size") != [512, 512]:
            raise VisionBackendError("D-FINE eval_spatial_size 漂移。")
        if resolved.get("DFINEPostProcessor", {}).get("num_top_queries") != 300:
            raise VisionBackendError("D-FINE postprocessor top-query contract 漂移。")
        resolved["tuning"] = None
        upstream_base = self.dfine_root / "configs" / "dfine" / "dfine_hgnetv2_l_coco.yml"
        if not upstream_base.is_file():
            raise VisionBackendError(f"D-FINE upstream base config 缺失：{upstream_base}")
        try:
            cfg = YAMLConfig(str(upstream_base), **resolved)
            model = cfg.model.to(self._device).eval()
            postprocessor = cfg.postprocessor.to(self._device).eval()
            state = torch.load(self.detector_checkpoint, map_location="cpu", weights_only=True)
            ema = state.get("ema")
            if not isinstance(ema, dict) or "module" not in ema:
                raise VisionBackendError("D-FINE checkpoint 不含冻结 EMA module。")
            incompatible = model.load_state_dict(ema["module"], strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise VisionBackendError(f"D-FINE EMA state dict 不兼容：{incompatible}")
        except VisionBackendError:
            raise
        except Exception as exc:
            raise VisionBackendError(f"D-FINE 初始化失败：{exc}") from exc
        self._detector, self._postprocessor = model, postprocessor
        del state, cfg
        return model, postprocessor

    def _classifier_infer(
        self, image: ValidatedImage
    ) -> tuple[dict[str, float], ClassifierClass | None, bool]:
        torch = self._torch()
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise VisionBackendError("rank03 预处理需要 torchvision。") from exc
        cfg = self.config["classifier"]
        transform = transforms.Compose(
            [
                transforms.Resize((int(cfg["input_size"]), int(cfg["input_size"])), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=tuple(cfg["normalization_mean"]),
                    std=tuple(cfg["normalization_std"]),
                ),
            ]
        )
        tensor = transform(image.image).unsqueeze(0).to(self._device)
        if self._device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        model = self._load_classifier()
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self._device.type == "cuda" and bool(cfg["amp"])
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits = model(tensor)
            values = torch.softmax(logits.float(), dim=1)[0].detach().cpu().tolist()
        if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
            raise VisionBackendError("ConvNeXt 产生了无效概率。")
        order = list(cfg["probability_order"])
        probabilities = {name: float(value) for name, value in zip(order, values, strict=True)}
        if abs(sum(probabilities.values()) - 1.0) > 1e-5:
            raise VisionBackendError("ConvNeXt 概率和不为 1。")
        predicted_class, argmax_tied = _native_classifier_argmax(probabilities, order)
        return probabilities, predicted_class, argmax_tied

    def _detector_infer(self, image: ValidatedImage) -> list[DetectionEvidence]:
        torch = self._torch()
        try:
            from torchvision.transforms import functional as F
        except ImportError as exc:
            raise VisionBackendError("rank03 预处理需要 torchvision。") from exc
        detector_cfg = self.config["detector"]
        resized = F.resize(
            image.image,
            [int(detector_cfg["input_size"]), int(detector_cfg["input_size"])],
            antialias=True,
        )
        tensor = F.pil_to_tensor(resized).float().div(255.0).unsqueeze(0).to(self._device)
        model, postprocessor = self._load_detector()
        original_size = torch.tensor(
            [[image.width, image.height]], dtype=torch.float32, device=self._device
        )
        with torch.inference_mode():
            outputs = model(tensor)
            result = postprocessor(outputs, original_size)[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()
        floor = float(detector_cfg["export_floor"])
        detections: list[DetectionEvidence] = []
        for raw_box, raw_score, raw_label in zip(boxes, scores, labels, strict=True):
            score = float(raw_score)
            if not math.isfinite(score) or score < floor:
                continue
            if int(raw_label) != int(detector_cfg["native_label"]):
                raise VisionBackendError("D-FINE 产生了非零原生标签。")
            if len(raw_box) != 4 or not all(math.isfinite(float(v)) for v in raw_box):
                continue
            x1, y1, x2, y2 = (float(v) for v in raw_box)
            x1, x2 = max(0.0, x1), min(float(image.width), x2)
            y1, y2 = max(0.0, y1), min(float(image.height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                DetectionEvidence(
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=min(1.0, max(0.0, score)),
                    label="tb_lesion_candidate",
                )
            )
        return detections

    def infer(self, *, case_id: str, image: ValidatedImage) -> VisionEvidence:
        started = time.perf_counter()
        try:
            with self._lock:
                probabilities, predicted_class, argmax_tied = self._classifier_infer(image)
            # Localization is an Agent tool.  Initial screening deliberately records
            # that it was not requested instead of paying for a hidden detector call.
            detections: list[DetectionEvidence] = []
            decision_fields = _policy_decision_fields(
                self.policy,
                probabilities=probabilities,
                predicted_class=predicted_class,
                argmax_tied=argmax_tied,
                detections=detections,
            )
            quality_status = image.quality_status
            if (image.width, image.height) != (512, 512):
                quality_status = "warning"
            return VisionEvidence(
                run_id=str(uuid.uuid4()),
                case_id=case_id,
                image_sha256=image.sha256,
                image_quality_status=quality_status,
                image_quality_codes=list(image.quality_warnings),
                image_source_format=image.source_format,
                input_transform_id=image.input_transform_id,
                image_width=image.width,
                image_height=image.height,
                classifier_model_id=str(self.config["model_bundle_id"]) + ":convnext_tiny",
                classifier_checkpoint_sha256=self.config["classifier"]["checkpoint_sha256"],
                class_probability_order=list(self.config["classifier"]["probability_order"]),
                class_probabilities=probabilities,
                classifier_decision_rule=decision_fields["classifier_decision_rule"],
                predicted_class=decision_fields["predicted_class"],
                classifier_argmax_tied=decision_fields["classifier_argmax_tied"],
                classifier_threshold=decision_fields["classifier_threshold"],
                classifier_flagged=decision_fields["classifier_flagged"],
                detector_model_id=str(self.config["model_bundle_id"]) + ":dfine_l",
                detector_checkpoint_sha256=self.config["detector"]["checkpoint_sha256"],
                detector_decision_role=decision_fields["detector_decision_role"],
                detector_threshold=decision_fields["detector_threshold"],
                detections=detections,
                detector_flagged=decision_fields["detector_flagged"],
                preprocessing_version="rank03-frozen-official-submission-v1",
                threshold_config_version=str(self.policy["policy_id"]),
                runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
                artifact_refs=[
                    f"device:{self._device.type}",
                    "classifier_precision:fp16_autocast"
                    if self._device.type == "cuda"
                    else "classifier_precision:fp32_cpu",
                    f"classifier_decision:{decision_fields['classifier_decision_rule']}",
                    f"detector_role:{decision_fields['detector_decision_role']}",
                    "detector_execution:not_requested",
                    "input_domain_deviation"
                    if (image.width, image.height) != (512, 512)
                    else "input_domain:tbx11k_512x512",
                ],
            )
        except VisionBackendError:
            raise
        except Exception as exc:
            raise VisionBackendError("rank03 模型加载或前向推理失败。") from exc

    def localize(
        self, *, case_id: str, image: ValidatedImage
    ) -> list[DetectionEvidence]:
        del case_id
        try:
            with self._lock:
                return self._detector_infer(image)
        except VisionBackendError:
            raise
        except Exception as exc:
            raise VisionBackendError("D-FINE 定位推理失败。") from exc
