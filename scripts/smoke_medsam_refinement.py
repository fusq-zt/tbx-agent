"""Run one real, non-patient MedSAM box-prompt engineering smoke.

The output is a technical runtime receipt, not a segmentation metric or
clinical-validation result.  It deliberately uses a generated raster and
generated lung masks, and writes no model weight into the source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tbx_agent.artifacts import ArtifactManager, load_manifest  # noqa: E402
from tbx_agent.vision.anatomy import (  # noqa: E402
    AnatomyEvidence,
    AnatomyMask,
    LungSide,
    build_generation_key,
    encode_binary_mask,
    evaluate_lung_masks,
)
from tbx_agent.vision.image_validator import ValidatedImage  # noqa: E402
from tbx_agent.vision.refinement import (  # noqa: E402
    HFMedSAMBoxRefinementBackend,
    HFMedSAMConfig,
)

SYNTHETIC_SEED = 0
SYNTHETIC_SPLIT_HASH = hashlib.sha256(
    b"tbx-medsam-synthetic-engineering-smoke-v2-single-generated-raster"
).hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    paths = [
        PROJECT_ROOT / "configs" / "model_sources.yaml",
        PROJECT_ROOT / "src" / "tbx_agent" / "vision" / "refinement" / "base.py",
        PROJECT_ROOT / "src" / "tbx_agent" / "vision" / "refinement" / "models.py",
        PROJECT_ROOT / "src" / "tbx_agent" / "vision" / "refinement" / "medsam_hf.py",
        Path(__file__).resolve(),
    ]
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_input() -> tuple[ValidatedImage, AnatomyEvidence]:
    width = height = 256
    raster = Image.new("RGB", (width, height), color=(210, 210, 210))
    draw = ImageDraw.Draw(raster)
    draw.ellipse((25, 18, 231, 250), fill=(170, 170, 170))
    draw.ellipse((35, 35, 116, 226), fill=(70, 70, 70))
    draw.ellipse((140, 35, 221, 226), fill=(70, 70, 70))
    draw.rectangle((119, 45, 137, 220), fill=(135, 135, 135))
    for y in range(52, 218, 24):
        draw.arc((28, y - 20, 228, y + 20), 190, 350, fill=(145, 145, 145), width=2)
    buffer = io.BytesIO()
    raster.save(buffer, format="PNG", optimize=False, compress_level=6)
    payload = buffer.getvalue()
    image_hash = hashlib.sha256(payload).hexdigest()
    image = ValidatedImage(
        image=raster,
        sha256=image_hash,
        width=width,
        height=height,
        source_format="PNG",
        input_transform_id="synthetic-engineering-raster-v1",
        quality_status="transport_valid",
        quality_warnings=("synthetic_non_patient_engineering_input",),
        artifact_bytes=payload,
    )
    left = [[False] * width for _ in range(height)]
    right = [[False] * width for _ in range(height)]
    for y in range(35, 227):
        for x in range(140, 222):
            left[y][x] = True
        for x in range(35, 117):
            right[y][x] = True
    anatomy_weight = hashlib.sha256(b"synthetic-pspnet-engineering-weight").hexdigest()
    anatomy_key = build_generation_key(
        image_sha256=image_hash,
        model_weight_sha256=anatomy_weight,
        preprocessing_id="synthetic-source-mask-v1",
        policy_id="paired-lung-qc-v1",
        backend_id="synthetic-lung-mask-fixture",
    )
    anatomy = AnatomyEvidence(
        run_id="synthetic-engineering-anatomy-run",
        case_id="synthetic-engineering-case",
        image_sha256=image_hash,
        image_width=width,
        image_height=height,
        backend_id="synthetic-lung-mask-fixture",
        model_id="synthetic-lung-mask-fixture",
        model_weight_sha256=anatomy_weight,
        model_state_dict_sha256=anatomy_weight,
        preprocessing_id="synthetic-source-mask-v1",
        policy_id="paired-lung-qc-v1",
        generation_key=anatomy_key,
        masks=[
            AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
            AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right)),
        ],
        qc=evaluate_lung_masks(
            left,
            right,
            source_width=width,
            source_height=height,
        ),
        runtime_ms=0,
    )
    return image, anatomy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--supersedes-receipt",
        type=Path,
        help="Optional immutable earlier receipt retained for audit history.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    manifest = load_manifest(PROJECT_ROOT / "configs" / "model_sources.yaml")
    manager = ArtifactManager(manifest, cache_dir=args.cache_dir)
    weight = manifest.artifacts["medsam_vit_base_weights"]
    config = manifest.artifacts["medsam_vit_base_config"]
    processor = manifest.artifacts["medsam_vit_base_preprocessor"]
    artifact_paths = [manager.path_for(item) for item in (weight, config, processor)]
    artifact_dir = artifact_paths[0].parent
    receipt: dict[str, object] = {
        "schema_version": 2,
        "run_kind": "medsam_real_forward_engineering_smoke",
        "hypothesis": (
            "The exact pinned MedSAM artifacts can execute one real box-prompt forward "
            "on a generated non-patient raster and produce source-space, routing-neutral evidence."
        ),
        "clinical_validation": False,
        "medical_performance_evaluation": False,
        "input_kind": "generated_non_patient_raster_and_masks",
        "seed": SYNTHETIC_SEED,
        "split_hash": SYNTHETIC_SPLIT_HASH,
        "selection_use": "none",
        "locked_local_test_used": False,
        "official_hidden_test_used": False,
        "started_at": started_at.isoformat(),
        "device_requested": args.device,
        "source_revision": {
            "repository_commit": _git_revision(),
            "source_digest": _source_digest(),
            "note": "source digest is authoritative for this uncommitted engineering run",
        },
        "model_revision": weight.revision,
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "path": str(path),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "verification_state": manager.verify(item).state,
            }
            for item, path in zip((weight, config, processor), artifact_paths, strict=True)
        ],
    }
    if args.supersedes_receipt is not None:
        receipt["supersedes"] = {
            "path": str(args.supersedes_receipt.resolve()),
            "sha256": _file_sha256(args.supersedes_receipt.resolve()),
            "preserved": True,
        }
    exit_code = 1
    try:
        if len({path.parent for path in artifact_paths}) != 1:
            raise RuntimeError("MedSAM artifacts do not share one immutable model directory")
        if any(item["verification_state"] != "valid" for item in receipt["artifacts"]):
            raise RuntimeError("one or more MedSAM artifacts failed verification")
        backend = HFMedSAMBoxRefinementBackend(
            config=HFMedSAMConfig(
                artifact_dir=artifact_dir,
                expected_weight_sha256=weight.sha256,
                expected_config_sha256=config.sha256,
                expected_preprocessor_sha256=processor.sha256,
                model_revision=weight.revision,
                device=args.device,
            )
        )
        image, anatomy = _synthetic_input()
        boxes = [(150.0, 55.0, 210.0, 175.0)]
        receipt["full_configuration"] = {
            "input": {
                "generator": "deterministic_non_patient_chest_like_raster_v1",
                "seed": SYNTHETIC_SEED,
                "width": image.width,
                "height": image.height,
                "sha256": image.sha256,
                "input_transform_id": image.input_transform_id,
            },
            "selection": {
                "split_hash": SYNTHETIC_SPLIT_HASH,
                "selection_use": "none",
                "locked_local_test_used": False,
                "official_hidden_test_used": False,
            },
            "detector_prompt_fixture": {
                "source": "fixed_dfine_style_xyxy_source_pixels",
                "boxes": boxes,
            },
            "anatomy_fixture": {
                "source": "generated_lung_masks_not_pspnet_inference",
                "generation_key": anatomy.generation_key,
                "qc": anatomy.qc.model_dump(mode="json"),
            },
            "medsam": {
                "backend_id": backend.backend_id,
                "model_id": backend.config.model_id,
                "model_revision": backend.config.model_revision,
                "preprocessing_id": backend.config.preprocessing_id,
                "device": backend.config.device,
                "policy": backend.policy.generation_parameters(),
            },
        }
        evidence = backend.refine(
            case_id=anatomy.case_id,
            image=image,
            boxes=boxes,
            anatomy=anatomy,
        )
        try:
            import torch

            peak_vram_bytes = (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda") and torch.cuda.is_available()
                else 0
            )
            torch_version = torch.__version__
        except ImportError:
            peak_vram_bytes = 0
            torch_version = "unavailable"
        item = evidence.items[0]
        receipt.update(
            {
                "status": "passed",
                "backend_id": evidence.backend_id,
                "model_state_dict_sha256": evidence.model_state_dict_sha256,
                "generation_key": evidence.generation_key,
                "detector_box_digest": evidence.detector_box_digest,
                "anatomy_mask_digest": evidence.anatomy_mask_digest,
                "prompt_count": len(evidence.items),
                "output_statuses": [entry.status.value for entry in evidence.items],
                "source_mask_shape": (
                    [item.mask.height, item.mask.width] if item.mask is not None else None
                ),
                "source_mask_pixels": (
                    item.mask.foreground_pixels if item.mask is not None else 0
                ),
                "metrics": {
                    "technical_prompt_coverage": len(evidence.items),
                    "output_statuses": [
                        entry.status.value for entry in evidence.items
                    ],
                    "source_mask_shape": (
                        [item.mask.height, item.mask.width]
                        if item.mask is not None
                        else None
                    ),
                    "source_mask_pixels": (
                        item.mask.foreground_pixels if item.mask is not None else 0
                    ),
                    "clinical_metrics": None,
                },
                "runtime": {
                    "backend_runtime_ms": evidence.runtime_ms,
                    "peak_vram_bytes": peak_vram_bytes,
                    "torch_version": torch_version,
                },
                "peak_vram_bytes": peak_vram_bytes,
                "torch_version": torch_version,
                "routing_effect": evidence.routing_effect,
            }
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - receipt keeps a stable failure record
        receipt.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    receipt["wall_runtime_ms"] = max(0, int((time.perf_counter() - started) * 1000))
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
