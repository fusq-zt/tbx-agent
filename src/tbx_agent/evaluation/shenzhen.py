from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from ..config import Settings
from ..schemas import VisualResult
from ..service import TBXAgentService
from ..vision.base import VisionBackend

DEFAULT_CONFIG_NAME = "shenzhen_external_eval_config.json"
DEFAULT_OUTPUT_DIRECTORY = "external_evaluation_runs"
ACTIVE_FUSION_POLICY_FILENAME = "fusion_policy.json"
ARGMAX_FUSION_POLICY_FILENAME = "fusion_policy_argmax_v2.json"
ARGMAX_FUSION_POLICY_ID = "rank03-agent-screening-demo-cls-argmax-det-advisory-v2"
LEGACY_SHENZHEN_EVALUATION_ID = "shenzhen-external-rank03-observational-v1"
LEGACY_SHENZHEN_FUSION_POLICY_FILENAME = "fusion_policy_sens98_legacy.json"
LEGACY_SHENZHEN_FUSION_POLICY_ID = "rank03-agent-screening-demo-official-val-sens98-v1"
METADATA_READ_LIMIT_BYTES = 1024 * 1024
SOURCE_HASH_SUFFIXES = {".json", ".jsonl", ".py", ".yaml", ".yml"}
FORBIDDEN_LOCALIZATION_METRICS = {
    "ap",
    "ap50",
    "ap50:95",
    "average_precision",
    "iou",
    "lesion_recall",
    "map",
}


class ExternalEvaluationContractError(RuntimeError):
    """Raised when the external dataset or its pre-registered contract drifts."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_tree_sha256(project_root: Path) -> str:
    """Hash runtime source/configs without hashing datasets or the mutable ledger."""

    candidates = [project_root / "pyproject.toml"]
    candidates.extend(sorted((project_root / "src").rglob("*.py")))
    for directory in ("configs", "knowledge", "evaluation"):
        root = project_root / directory
        candidates.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in SOURCE_HASH_SUFFIXES
            and path.name != "ledger.jsonl"
        )

    manifest_path = project_root / "knowledge" / "source_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        knowledge_root = (project_root / "knowledge").resolve()
        for source in manifest.get("sources", []):
            local_path = source.get("local_path")
            if not isinstance(local_path, str):
                continue
            source_path = (knowledge_root / local_path).resolve()
            source_path.relative_to(knowledge_root)
            if source_path.is_file():
                candidates.append(source_path)

    digest = hashlib.sha256()
    seen: set[Path] = set()
    for path in sorted(candidates):
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        relative = resolved.relative_to(project_root.resolve()).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_sha256(value: Any, field: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ExternalEvaluationContractError(f"{field} must be a complete SHA256")
    return normalized


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("selection_use") is not False:
        raise ExternalEvaluationContractError("selection_use must be false")
    if config.get("threshold_selection") is not False:
        raise ExternalEvaluationContractError("threshold_selection must be false")
    if config.get("calibration") is not False:
        raise ExternalEvaluationContractError("calibration must be false")
    if config.get("external_test_only") is not True:
        raise ExternalEvaluationContractError("external_test_only must be true")
    if config.get("locked_or_hidden_test_used") is not False:
        raise ExternalEvaluationContractError("locked_or_hidden_test_used must be false")
    if config.get("official_hidden_test_used") is not False:
        raise ExternalEvaluationContractError("official_hidden_test_used must be false")
    if not isinstance(config.get("seed"), int):
        raise ExternalEvaluationContractError("seed must be an integer")
    changed = config.get("major_variables_changed")
    if not isinstance(changed, list) or not changed or len(set(changed)) > 2:
        raise ExternalEvaluationContractError(
            "major_variables_changed must contain one or two unique variables"
        )
    forbidden_knobs = {
        "classifier_rule",
        "classifier_threshold",
        "detector_threshold",
        "fusion_policy_filename",
        "limit",
        "resize",
        "subset",
    }
    found_forbidden = forbidden_knobs.intersection(_walk_keys(config))
    if found_forbidden:
        raise ExternalEvaluationContractError(
            f"external evaluation config exposes forbidden tuning knobs: {sorted(found_forbidden)}"
        )

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ExternalEvaluationContractError("dataset configuration is required")
    if dataset.get("bbox_ground_truth_available") is not False:
        raise ExternalEvaluationContractError(
            "Shenzhen evaluation must declare bbox_ground_truth_available=false"
        )
    for field in (
        "expected_archive_sha256",
        "expected_metadata_sha256",
        "expected_split_manifest_sha256",
    ):
        _validate_sha256(dataset.get(field), f"dataset.{field}")
    for field in ("expected_archive_bytes", "expected_image_count"):
        if not isinstance(dataset.get(field), int) or int(dataset[field]) <= 0:
            raise ExternalEvaluationContractError(f"dataset.{field} must be positive")
    expected_counts = dataset.get("expected_label_counts")
    if not isinstance(expected_counts, dict) or set(expected_counts) != {"0", "1"}:
        raise ExternalEvaluationContractError(
            "dataset.expected_label_counts must contain labels '0' and '1'"
        )
    if any(not isinstance(value, int) or value < 0 for value in expected_counts.values()):
        raise ExternalEvaluationContractError("expected label counts must be non-negative")
    if sum(expected_counts.values()) != dataset["expected_image_count"]:
        raise ExternalEvaluationContractError("expected label counts do not match image count")
    try:
        pattern = re.compile(str(dataset["study_id_pattern"]))
    except (KeyError, re.error) as exc:
        raise ExternalEvaluationContractError("dataset.study_id_pattern is invalid") from exc
    if "label" not in pattern.groupindex:
        raise ExternalEvaluationContractError(
            "dataset.study_id_pattern must include a named 'label' group"
        )
    if not str(dataset.get("image_entry_prefix", "")):
        raise ExternalEvaluationContractError("dataset.image_entry_prefix is required")
    if not str(dataset.get("metadata_entry", "")):
        raise ExternalEvaluationContractError("dataset.metadata_entry is required")

    configured_metrics = {
        str(item).strip().lower() for item in config.get("metrics", []) if str(item).strip()
    }
    prohibited = configured_metrics.intersection(FORBIDDEN_LOCALIZATION_METRICS)
    if prohibited:
        raise ExternalEvaluationContractError(
            f"bbox-free dataset cannot request localization metrics: {sorted(prohibited)}"
        )
    bootstrap = config.get("confidence_intervals", {})
    if bootstrap.get("method") != "stratified_bootstrap_percentile_95":
        raise ExternalEvaluationContractError("confidence interval method is not pre-registered")
    if not isinstance(bootstrap.get("replicates"), int) or bootstrap["replicates"] <= 0:
        raise ExternalEvaluationContractError("bootstrap replicates must be positive")
    return config


def _fusion_policy_binding(config: dict[str, Any]) -> tuple[str, str]:
    """Bind a governed evaluation identity to one exact policy identity.

    The original Shenzhen run predates the active argmax policy.  Its evaluation
    ID therefore always resolves to the immutable legacy policy snapshot, even
    when the application's active policy changes.  New evaluation identities
    must state the active policy ID they expect and cannot select a policy file.
    """

    evaluation_id = str(config.get("evaluation_id", "")).strip()
    if evaluation_id == LEGACY_SHENZHEN_EVALUATION_ID:
        return (
            LEGACY_SHENZHEN_FUSION_POLICY_FILENAME,
            LEGACY_SHENZHEN_FUSION_POLICY_ID,
        )
    expected_policy_id = str(config.get("expected_fusion_policy_id", "")).strip()
    if not expected_policy_id:
        raise ExternalEvaluationContractError(
            "non-legacy evaluation must declare expected_fusion_policy_id"
        )
    if expected_policy_id == ARGMAX_FUSION_POLICY_ID:
        return ARGMAX_FUSION_POLICY_FILENAME, expected_policy_id
    return ACTIVE_FUSION_POLICY_FILENAME, expected_policy_id


def _policy_decision_contract(policy: dict[str, Any]) -> tuple[str, str]:
    """Translate a policy document into the runtime decision schema it requires."""

    classifier_rule = policy.get("classifier_rule")
    if classifier_rule == "native_three_class_argmax":
        if policy.get("detector_role") != "advisory_localization_only":
            raise ExternalEvaluationContractError(
                "native argmax policy must declare detector_role=advisory_localization_only"
            )
        if "classifier_threshold" in policy or "detector_threshold" in policy:
            raise ExternalEvaluationContractError(
                "native argmax/advisory policy must not contain decision thresholds"
            )
        return "native_three_class_argmax", "advisory_localization_only"
    if classifier_rule == "p_tb_gte_threshold":
        if policy.get("detector_role") == "advisory_localization_only":
            threshold = policy.get("classifier_threshold")
            if (
                not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not math.isfinite(float(threshold))
                or not 0.0 <= float(threshold) <= 1.0
                or "detector_threshold" in policy
                or "detector_rule" in policy
            ):
                raise ExternalEvaluationContractError(
                    "advisory p_tb threshold policy has an invalid decision contract"
                )
            return "p_tb_gte_threshold", "advisory_localization_only"
        if policy.get("detector_rule") != "max_category_agnostic_score_gte_threshold":
            raise ExternalEvaluationContractError(
                "legacy threshold policy must declare its detector threshold rule"
            )
        if not isinstance(policy.get("classifier_threshold"), (int, float)) or not isinstance(
            policy.get("detector_threshold"), (int, float)
        ):
            raise ExternalEvaluationContractError(
                "legacy threshold policy must contain classifier and detector thresholds"
            )
        return "legacy_p_tb_threshold", "legacy_vote"
    raise ExternalEvaluationContractError(
        f"unsupported governed classifier rule: {classifier_rule!r}"
    )


def _read_zip_entry_bounded(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
) -> bytes:
    if info.file_size > max_bytes:
        raise ExternalEvaluationContractError(
            f"ZIP entry exceeds bounded read limit: {info.filename}"
        )
    with archive.open(info, "r") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ExternalEvaluationContractError(
            f"ZIP entry exceeded bounded read limit while streaming: {info.filename}"
        )
    if len(payload) != info.file_size:
        raise ExternalEvaluationContractError(f"ZIP entry size drift: {info.filename}")
    return payload


def _safe_zip_name(name: str) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and ":" not in path.parts[0]


def _canonical_split_manifest(samples: list[dict[str, Any]]) -> tuple[str, int]:
    canonical = [
        {
            "crc32": sample["crc32"],
            "entry_name": sample["entry_name"],
            "label": sample["label"],
            "study_id": sample["study_id"],
            "uncompressed_size": sample["uncompressed_size"],
        }
        for sample in sorted(samples, key=lambda item: item["study_id"])
    ]
    payload = (
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return _sha256_bytes(payload), len(payload)


def _audit_archive(
    archive_path: Path,
    *,
    config: dict[str, Any],
    max_image_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = config["dataset"]
    if not archive_path.is_file():
        raise ExternalEvaluationContractError(f"archive is missing: {archive_path}")
    archive_bytes = archive_path.stat().st_size
    if archive_bytes != int(dataset["expected_archive_bytes"]):
        raise ExternalEvaluationContractError(
            f"archive byte size drift: {archive_bytes} != {dataset['expected_archive_bytes']}"
        )

    hash_started = time.perf_counter()
    archive_sha256 = _sha256_file(archive_path)
    hash_seconds = time.perf_counter() - hash_started
    expected_archive_sha256 = _validate_sha256(
        dataset["expected_archive_sha256"], "dataset.expected_archive_sha256"
    )
    if archive_sha256 != expected_archive_sha256:
        raise ExternalEvaluationContractError(
            f"archive SHA256 drift: {archive_sha256} != {expected_archive_sha256}"
        )

    audit_started = time.perf_counter()
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExternalEvaluationContractError("archive is not a readable ZIP") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ExternalEvaluationContractError(f"duplicate ZIP names: {duplicates[:5]}")
        unsafe = [info.filename for info in infos if not _safe_zip_name(info.filename)]
        if unsafe:
            raise ExternalEvaluationContractError(f"unsafe ZIP names: {unsafe[:5]}")
        encrypted = [info.filename for info in infos if info.flag_bits & 0x1]
        if encrypted:
            raise ExternalEvaluationContractError(f"encrypted ZIP entries: {encrypted[:5]}")
        symlinks = [
            info.filename for info in infos if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
        ]
        if symlinks:
            raise ExternalEvaluationContractError(f"symlink ZIP entries: {symlinks[:5]}")

        metadata_entry = str(dataset["metadata_entry"])
        try:
            metadata_info = archive.getinfo(metadata_entry)
        except KeyError as exc:
            raise ExternalEvaluationContractError("metadata entry is missing") from exc
        metadata_payload = _read_zip_entry_bounded(
            archive,
            metadata_info,
            max_bytes=METADATA_READ_LIMIT_BYTES,
        )
        metadata_sha256 = _sha256_bytes(metadata_payload)
        expected_metadata_sha256 = _validate_sha256(
            dataset["expected_metadata_sha256"], "dataset.expected_metadata_sha256"
        )
        if metadata_sha256 != expected_metadata_sha256:
            raise ExternalEvaluationContractError(
                f"metadata SHA256 drift: {metadata_sha256} != {expected_metadata_sha256}"
            )
        try:
            rows = list(
                csv.DictReader(io.StringIO(metadata_payload.decode("utf-8-sig", errors="strict")))
            )
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ExternalEvaluationContractError("metadata CSV is invalid") from exc
        required_columns = {"study_id", "findings"}
        observed_columns = set(rows[0]) if rows else set()
        if not required_columns.issubset(observed_columns):
            raise ExternalEvaluationContractError("metadata CSV is missing required columns")

        prefix = str(dataset["image_entry_prefix"])
        pattern = re.compile(str(dataset["study_id_pattern"]))
        image_infos = [
            info
            for info in infos
            if not info.is_dir()
            and info.filename.startswith(prefix)
            and info.filename.lower().endswith(".png")
        ]
        allowed_names = {metadata_entry, *(info.filename for info in image_infos)}
        extras = [
            info.filename
            for info in infos
            if not info.is_dir() and info.filename not in allowed_names
        ]
        if extras:
            raise ExternalEvaluationContractError(f"unexpected ZIP entries: {extras[:5]}")
        if len(image_infos) != int(dataset["expected_image_count"]):
            raise ExternalEvaluationContractError(
                f"image count drift: {len(image_infos)} != {dataset['expected_image_count']}"
            )

        by_basename: dict[str, zipfile.ZipInfo] = {}
        for info in image_infos:
            basename = PurePosixPath(info.filename).name
            if info.filename != prefix + basename:
                raise ExternalEvaluationContractError(
                    f"unexpected nested image path: {info.filename}"
                )
            if basename in by_basename:
                raise ExternalEvaluationContractError(f"duplicate image basename: {basename}")
            if info.file_size > max_image_bytes:
                raise ExternalEvaluationContractError(
                    f"image exceeds service upload limit: {info.filename}"
                )
            by_basename[basename] = info

        study_ids = [str(row.get("study_id", "")).strip() for row in rows]
        duplicate_studies = sorted(
            study_id for study_id, count in Counter(study_ids).items() if count > 1
        )
        if duplicate_studies:
            raise ExternalEvaluationContractError(
                f"duplicate metadata study IDs: {duplicate_studies[:5]}"
            )
        if set(study_ids) != set(by_basename):
            missing = sorted(set(study_ids) - set(by_basename))
            unreferenced = sorted(set(by_basename) - set(study_ids))
            raise ExternalEvaluationContractError(
                f"metadata/image mismatch; missing={missing[:5]}, unreferenced={unreferenced[:5]}"
            )

        samples: list[dict[str, Any]] = []
        negative_finding = str(dataset.get("negative_finding", "normal")).strip().casefold()
        for row in rows:
            study_id = str(row["study_id"]).strip()
            match = pattern.fullmatch(study_id)
            if match is None:
                raise ExternalEvaluationContractError(f"invalid study ID: {study_id}")
            label = int(match.group("label"))
            if label not in {0, 1}:
                raise ExternalEvaluationContractError(f"invalid binary label: {study_id}")
            finding = str(row.get("findings", "")).strip().casefold()
            if (label == 0) != (finding == negative_finding):
                raise ExternalEvaluationContractError(
                    f"filename label and metadata finding disagree: {study_id}"
                )
            info = by_basename[study_id]
            samples.append(
                {
                    "study_id": study_id,
                    "label": label,
                    "entry_name": info.filename,
                    "crc32": f"{info.CRC:08x}",
                    "uncompressed_size": info.file_size,
                }
            )

        samples.sort(key=lambda item: item["study_id"])
        label_counts = Counter(str(sample["label"]) for sample in samples)
        expected_label_counts = {
            str(key): int(value) for key, value in dataset["expected_label_counts"].items()
        }
        if dict(sorted(label_counts.items())) != dict(sorted(expected_label_counts.items())):
            raise ExternalEvaluationContractError(
                f"label count drift: {dict(label_counts)} != {expected_label_counts}"
            )
        split_hash, split_manifest_bytes = _canonical_split_manifest(samples)
        expected_split = _validate_sha256(
            dataset["expected_split_manifest_sha256"],
            "dataset.expected_split_manifest_sha256",
        )
        if split_hash != expected_split:
            raise ExternalEvaluationContractError(
                f"split manifest drift: {split_hash} != {expected_split}"
            )

    audit_seconds = time.perf_counter() - audit_started
    audit = {
        "archive_path": str(archive_path.resolve()),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "archive_hash_seconds": hash_seconds,
        "metadata_entry": metadata_entry,
        "metadata_sha256": metadata_sha256,
        "image_count": len(samples),
        "label_counts": dict(sorted(label_counts.items())),
        "split_manifest_sha256": split_hash,
        "split_manifest_bytes": split_manifest_bytes,
        "split_definition": "all metadata-linked images sorted by study_id; no sampling",
        "max_image_bytes": max(sample["uncompressed_size"] for sample in samples),
        "encrypted_entry_count": 0,
        "duplicate_entry_count": 0,
        "archive_audit_seconds": audit_seconds,
        "extraction_performed": False,
    }
    return audit, samples


class _ProcessVramSampler:
    def __init__(self, *, enabled: bool, interval_seconds: float = 1.0):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.peak_bytes = 0
        self.sample_count = 0
        self.matched_sample_count = 0
        self.available = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.enabled:
            self._thread.start()

    def _sample(self) -> bool:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError:
            self.error = "nvidia-smi unavailable"
            return False
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.error = type(exc).__name__
            return False
        if completed.returncode != 0:
            self.error = "nvidia-smi query failed"
            return False
        self.available = True
        self.sample_count += 1
        total_mib = 0
        matched = False
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.rsplit(",", 2)]
            if len(fields) != 3:
                continue
            try:
                observed_pid = int(fields[0])
                memory_mib = int(fields[2])
            except ValueError:
                continue
            if observed_pid != self.pid:
                continue
            matched = True
            total_mib += memory_mib
        if matched:
            self.matched_sample_count += 1
            self.peak_bytes = max(self.peak_bytes, total_mib * 1024 * 1024)
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sample():
                return
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        if self.enabled:
            self._stop.set()
            self._thread.join(timeout=5)
        return {
            "method": "nvidia-smi current-process used_gpu_memory polling",
            "enabled": self.enabled,
            "available": self.available,
            "sample_count": self.sample_count,
            "matched_sample_count": self.matched_sample_count,
            "peak_bytes": self.peak_bytes,
            "error": self.error,
        }


class _TorchCudaTracker:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.torch: Any = None
        self.device: Any = None
        self.error: str | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import torch

            self.torch = torch
            if not torch.cuda.is_available():
                self.error = "CUDA unavailable"
                return
            self.device = torch.device("cuda")
            torch.cuda.reset_peak_memory_stats(self.device)
        except (ImportError, RuntimeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> dict[str, Any]:
        if self.device is None or self.torch is None:
            return {
                "method": "torch.cuda peak allocator counters",
                "enabled": self.enabled,
                "available": False,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "error": self.error,
            }
        try:
            self.torch.cuda.synchronize(self.device)
            allocated = int(self.torch.cuda.max_memory_allocated(self.device))
            reserved = int(self.torch.cuda.max_memory_reserved(self.device))
            return {
                "method": "torch.cuda peak allocator counters",
                "enabled": self.enabled,
                "available": True,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "device": str(self.torch.cuda.get_device_name(self.device)),
                "error": None,
            }
        except RuntimeError as exc:
            return {
                "method": "torch.cuda peak allocator counters",
                "enabled": self.enabled,
                "available": False,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            name: _package_version(name)
            for name in ("tbx-agent", "pillow", "torch", "torchvision", "timm")
        },
    }


def _dimension_warnings(width: int | None, height: int | None, quality: str | None) -> list[str]:
    if width is None or height is None:
        return []
    warnings: list[str] = []
    if min(width, height) < 512:
        warnings.append("resolution_below_512")
    if (width, height) != (512, 512):
        warnings.append("outside_rank03_validated_512x512_domain")
    if not 0.65 <= width / height <= 1.35:
        warnings.append("unusual_aspect_ratio")
    if quality == "warning" and not warnings:
        warnings.append("validator_warning_not_exposed_by_case_schema")
    return warnings


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _successful_record(
    *,
    index: int,
    sample: dict[str, Any],
    payload_sha256: str,
    case: Any,
    response: Any,
    expected_safety_policy_id: str,
    expected_fusion_policy_id: str,
    expected_classifier_decision_rule: str,
    expected_detector_decision_role: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    evidence = case.vision_evidence
    fusion = case.fusion_decision
    visual_result = fusion.visual_result.value if fusion is not None else "technical_failure"
    scope_notice_present = any(
        "辅助筛查" in limitation and "确诊" in limitation for limitation in response.limitations
    )
    response_contract = {
        "response_kind": response.response_kind.value,
        "safety_policy_id": response.safety_policy_id,
        "narration_status": response.narration_status.value,
        "narrator_backend": response.narrator_backend,
        "scope_notice_present": scope_notice_present,
        "citation_count": len(response.citations),
    }
    response_contract["contract_pass"] = bool(
        response_contract["response_kind"] == "visual_screening_result"
        and response_contract["safety_policy_id"] == expected_safety_policy_id
        and response_contract["narration_status"] == "not_configured"
        and response_contract["narrator_backend"] is None
        and response_contract["scope_notice_present"]
    )
    predicted_class = (
        _enum_value(getattr(evidence, "predicted_class", None)) if evidence is not None else None
    )
    classifier_decision_rule = (
        getattr(evidence, "classifier_decision_rule", None) if evidence is not None else None
    )
    detector_decision_role = (
        getattr(evidence, "detector_decision_role", None) if evidence is not None else None
    )
    fusion_policy_id = fusion.policy_id if fusion is not None else None
    decision_contract = {
        "expected_fusion_policy_id": expected_fusion_policy_id,
        "actual_fusion_policy_id": fusion_policy_id,
        "expected_classifier_decision_rule": expected_classifier_decision_rule,
        "actual_classifier_decision_rule": classifier_decision_rule,
        "expected_detector_decision_role": expected_detector_decision_role,
        "actual_detector_decision_role": detector_decision_role,
    }
    decision_contract["contract_pass"] = bool(
        fusion_policy_id == expected_fusion_policy_id
        and classifier_decision_rule == expected_classifier_decision_rule
        and detector_decision_role == expected_detector_decision_role
    )
    vision_technical = visual_result == VisualResult.TECHNICAL_FAILURE.value or evidence is None
    technical = (
        vision_technical
        or not response_contract["contract_pass"]
        or not decision_contract["contract_pass"]
    )
    max_exported_score = (
        max((detection.score for detection in evidence.detections), default=None)
        if evidence is not None
        else None
    )
    max_score_for_flag = max_exported_score if max_exported_score is not None else 0.0
    quality = evidence.image_quality_status if evidence is not None else None
    record = {
        "record_index": index,
        "study_id": sample["study_id"],
        "ground_truth_label": sample["label"],
        "zip_entry": sample["entry_name"],
        "zip_crc32": sample["crc32"],
        "image_sha256": payload_sha256,
        "image_bytes": sample["uncompressed_size"],
        "image_width": case.image_width,
        "image_height": case.image_height,
        "source_format": "PNG",
        "image_quality_status": quality,
        "quality_warnings": _dimension_warnings(case.image_width, case.image_height, quality),
        "classifier": {
            "class_probability_order": (
                evidence.class_probability_order if evidence is not None else None
            ),
            "class_probabilities": evidence.class_probabilities if evidence is not None else None,
            "tb_probability": (
                evidence.class_probabilities.get("tb") if evidence is not None else None
            ),
            "decision_rule": classifier_decision_rule,
            "predicted_class": predicted_class,
            "argmax_tied": (
                getattr(evidence, "classifier_argmax_tied", False) if evidence is not None else None
            ),
            "threshold": (
                getattr(evidence, "classifier_threshold", None) if evidence is not None else None
            ),
            "flagged": (
                getattr(evidence, "classifier_flagged", None) if evidence is not None else None
            ),
            "model_id": evidence.classifier_model_id if evidence is not None else None,
            "checkpoint_sha256": (
                evidence.classifier_checkpoint_sha256 if evidence is not None else None
            ),
        },
        "detector": {
            "max_detector_score": max_score_for_flag if evidence is not None else None,
            "max_exported_candidate_score": max_exported_score,
            "exported_candidate_count": len(evidence.detections) if evidence is not None else None,
            "decision_role": detector_decision_role,
            "threshold": (
                getattr(evidence, "detector_threshold", None) if evidence is not None else None
            ),
            "flagged": (
                getattr(evidence, "detector_flagged", None) if evidence is not None else None
            ),
            "model_id": evidence.detector_model_id if evidence is not None else None,
            "checkpoint_sha256": (
                evidence.detector_checkpoint_sha256 if evidence is not None else None
            ),
        },
        "fusion": {
            "policy_id": fusion_policy_id,
            "visual_result": visual_result,
            "review_required": fusion.review_required if fusion is not None else False,
            "review_reasons": fusion.review_reasons if fusion is not None else ["missing_fusion"],
            "classifier_decision_rule": (
                getattr(fusion, "classifier_decision_rule", None) if fusion is not None else None
            ),
            "predicted_class": (
                _enum_value(getattr(fusion, "predicted_class", None))
                if fusion is not None
                else None
            ),
            "classifier_flagged": (
                getattr(fusion, "classifier_flagged", None) if fusion is not None else None
            ),
            "detector_decision_role": (
                getattr(fusion, "detector_decision_role", None) if fusion is not None else None
            ),
            "detector_flagged": (
                getattr(fusion, "detector_flagged", None) if fusion is not None else None
            ),
            "decision_contract": decision_contract,
            "clinical_validation": fusion.clinical_validation if fusion is not None else False,
        },
        "service": {
            "case_id": case.case_id,
            "response_visual_result": (
                response.visual_result.value if response.visual_result is not None else None
            ),
            "review_status": (
                response.review_status.value if response.review_status is not None else None
            ),
            "reused_existing_assessment": response.reused_existing_assessment,
            "response_contract": response_contract,
        },
        "technical_failure": {
            "occurred": technical,
            "stage": (
                "vision_backend"
                if vision_technical
                else "response_contract"
                if not response_contract["contract_pass"]
                else "decision_contract"
                if not decision_contract["contract_pass"]
                else None
            ),
            "error_type": (
                "VisionBackendError_sanitized_by_service"
                if vision_technical
                else "ResponseContractViolation"
                if not response_contract["contract_pass"]
                else "DecisionContractViolation"
                if not decision_contract["contract_pass"]
                else None
            ),
            "message": None,
        },
        "runtime": {
            "service_assessment_seconds": elapsed_seconds,
            "backend_runtime_ms": evidence.runtime_ms if evidence is not None else None,
        },
    }
    return record


def _exception_record(
    *,
    index: int,
    sample: dict[str, Any],
    payload_sha256: str | None,
    stage: str,
    exc: Exception,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "record_index": index,
        "study_id": sample["study_id"],
        "ground_truth_label": sample["label"],
        "zip_entry": sample["entry_name"],
        "zip_crc32": sample["crc32"],
        "image_sha256": payload_sha256,
        "image_bytes": sample["uncompressed_size"],
        "image_width": None,
        "image_height": None,
        "source_format": "PNG",
        "image_quality_status": "technical_failure",
        "quality_warnings": [],
        "classifier": {
            "class_probability_order": None,
            "class_probabilities": None,
            "tb_probability": None,
            "decision_rule": None,
            "predicted_class": None,
            "argmax_tied": None,
            "threshold": None,
            "flagged": None,
            "model_id": None,
            "checkpoint_sha256": None,
        },
        "detector": {
            "max_detector_score": None,
            "max_exported_candidate_score": None,
            "exported_candidate_count": None,
            "decision_role": None,
            "threshold": None,
            "flagged": None,
            "model_id": None,
            "checkpoint_sha256": None,
        },
        "fusion": {
            "policy_id": None,
            "visual_result": VisualResult.TECHNICAL_FAILURE.value,
            "review_required": False,
            "review_reasons": [f"{stage}_error"],
            "classifier_decision_rule": None,
            "predicted_class": None,
            "classifier_flagged": None,
            "detector_decision_role": None,
            "detector_flagged": None,
            "decision_contract": {
                "expected_fusion_policy_id": None,
                "actual_fusion_policy_id": None,
                "expected_classifier_decision_rule": None,
                "actual_classifier_decision_rule": None,
                "expected_detector_decision_role": None,
                "actual_detector_decision_role": None,
                "contract_pass": False,
            },
            "clinical_validation": False,
        },
        "service": {
            "case_id": None,
            "response_visual_result": None,
            "review_status": None,
            "reused_existing_assessment": False,
            "response_contract": {
                "response_kind": None,
                "safety_policy_id": None,
                "narration_status": None,
                "narrator_backend": None,
                "scope_notice_present": False,
                "citation_count": 0,
                "contract_pass": False,
            },
        },
        "technical_failure": {
            "occurred": True,
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        },
        "runtime": {
            "service_assessment_seconds": elapsed_seconds,
            "backend_runtime_ms": None,
        },
    }


def _point_binary_metrics(labels: list[int], predictions: list[bool]) -> dict[str, Any]:
    tp = sum(
        label == 1 and prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    tn = sum(
        label == 0 and not prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    fp = sum(
        label == 0 and prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    fn = sum(
        label == 1 and not prediction for label, prediction in zip(labels, predictions, strict=True)
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1_positive = ratio(2 * tp, 2 * tp + fp + fn)
    f1_negative = ratio(2 * tn, 2 * tn + fp + fn)
    balanced = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    macro_f1 = (
        (f1_positive + f1_negative) / 2
        if f1_positive is not None and f1_negative is not None
        else None
    )
    return {
        "evaluable_count": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": ratio(tp + tn, len(labels)),
        "balanced_accuracy": balanced,
        "positive_predictive_value": ratio(tp, tp + fp),
        "negative_predictive_value": ratio(tn, tn + fn),
        "f1_tb_positive": f1_positive,
        "binary_macro_f1": macro_f1,
        "predicted_positive_rate": ratio(tp + fp, len(labels)),
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _binary_metrics_with_ci(
    labels: list[int],
    predictions: list[bool],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    point = _point_binary_metrics(labels, predictions)
    metric_names = (
        "sensitivity",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "positive_predictive_value",
        "negative_predictive_value",
        "f1_tb_positive",
        "binary_macro_f1",
        "predicted_positive_rate",
    )
    positive_predictions = [
        prediction for label, prediction in zip(labels, predictions, strict=True) if label == 1
    ]
    negative_predictions = [
        prediction for label, prediction in zip(labels, predictions, strict=True) if label == 0
    ]
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = random.Random(seed)
    if labels and positive_predictions and negative_predictions:
        for _ in range(replicates):
            sampled_positive = [rng.choice(positive_predictions) for _ in positive_predictions]
            sampled_negative = [rng.choice(negative_predictions) for _ in negative_predictions]
            sampled_labels = [1] * len(sampled_positive) + [0] * len(sampled_negative)
            sampled_predictions = sampled_positive + sampled_negative
            observed = _point_binary_metrics(sampled_labels, sampled_predictions)
            for name in metric_names:
                value = observed[name]
                if value is not None:
                    samples[name].append(float(value))
    point["confidence_intervals_95"] = {
        "method": "stratified_bootstrap_percentile_95",
        "seed": seed,
        "replicates": replicates,
        "intervals": {
            name: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "valid_replicates": len(values),
            }
            for name, values in samples.items()
        },
    }
    return point


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _roc_auc_with_ci(
    labels: list[int],
    scores: list[float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    positive_scores = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    bootstrap_values: list[float] = []
    rng = random.Random(seed)
    if positive_scores and negative_scores:
        for _ in range(replicates):
            sampled_positive = [rng.choice(positive_scores) for _ in positive_scores]
            sampled_negative = [rng.choice(negative_scores) for _ in negative_scores]
            sampled_labels = [1] * len(sampled_positive) + [0] * len(sampled_negative)
            observed = _roc_auc(sampled_labels, sampled_positive + sampled_negative)
            if observed is not None:
                bootstrap_values.append(observed)
    return {
        "value": _roc_auc(labels, scores),
        "confidence_interval_95": {
            "method": "stratified_bootstrap_percentile_95",
            "seed": seed,
            "replicates": replicates,
            "lower": _percentile(bootstrap_values, 0.025),
            "upper": _percentile(bootstrap_values, 0.975),
            "valid_replicates": len(bootstrap_values),
        },
    }


def _wilson_interval(successes: int, total: int) -> dict[str, float | None]:
    if total == 0:
        return {"lower": None, "upper": None}
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "count": successes,
        "denominator": total,
        "value": successes / total if total else None,
        "confidence_interval_95": {
            "method": "Wilson score",
            **_wilson_interval(successes, total),
        },
    }


def _calculate_metrics(
    records: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    def classifier_is_evaluable(record: dict[str, Any]) -> bool:
        if record["technical_failure"]["occurred"]:
            return False
        classifier = record["classifier"]
        if classifier.get("decision_rule") == "native_three_class_argmax":
            return not classifier.get("argmax_tied", False) and classifier.get(
                "predicted_class"
            ) in {"healthy", "sick_non_tb", "tb"}
        return classifier.get("flagged") is not None

    def detector_is_evaluable(record: dict[str, Any]) -> bool:
        return (
            not record["technical_failure"]["occurred"]
            and record["detector"].get("flagged") is not None
        )

    classifier_records = [record for record in records if classifier_is_evaluable(record)]
    detector_records = [record for record in records if detector_is_evaluable(record)]
    detector_score_records = [
        record
        for record in records
        if not record["technical_failure"]["occurred"]
        and record["detector"].get("max_detector_score") is not None
    ]
    classifier_labels = [int(record["ground_truth_label"]) for record in classifier_records]
    classifier_predictions = [
        bool(record["classifier"]["flagged"]) for record in classifier_records
    ]
    detector_labels = [int(record["ground_truth_label"]) for record in detector_records]
    detector_predictions = [bool(record["detector"]["flagged"]) for record in detector_records]
    classifier_scores = [
        float(record["classifier"]["tb_probability"]) for record in classifier_records
    ]
    detector_scores = [
        float(record["detector"]["max_detector_score"]) for record in detector_score_records
    ]
    detector_score_labels = [int(record["ground_truth_label"]) for record in detector_score_records]
    final_labels = [int(record["ground_truth_label"]) for record in records]
    final_predictions = [
        record["technical_failure"]["occurred"]
        or record["fusion"]["visual_result"]
        not in {
            VisualResult.MODEL_NOT_FLAGGED.value,
            VisualResult.NON_TB_ABNORMAL.value,
        }
        for record in records
    ]
    total = len(records)
    result_counts = Counter(record["fusion"]["visual_result"] for record in records)
    technical_count = sum(record["technical_failure"]["occurred"] for record in records)
    native_decision_count = sum(
        not record["technical_failure"]["occurred"]
        and record["classifier"].get("decision_rule") == "native_three_class_argmax"
        for record in records
    )
    argmax_tie_count = sum(
        not record["technical_failure"]["occurred"]
        and record["classifier"].get("decision_rule") == "native_three_class_argmax"
        and bool(record["classifier"].get("argmax_tied"))
        for record in records
    )
    advisory_detector_count = sum(
        not record["technical_failure"]["occurred"]
        and record["detector"].get("decision_role") == "advisory_localization_only"
        for record in records
    )
    detector_metric_status = (
        "computed_legacy_detector_decision_rule"
        if detector_records
        else "not_computed_detector_advisory_only"
        if advisory_detector_count
        else "not_computed_no_evaluable_detector_decision"
    )
    classifier_evaluation_scope = (
        "TB versus non-TB from the frozen native three-class argmax rule; "
        "exact ties abstain to human review"
        if native_decision_count
        else "TB versus non-TB at the frozen legacy classifier flag threshold"
    )
    review_count = sum(record["fusion"]["review_required"] for record in records)
    quality_warning_count = sum(record["image_quality_status"] == "warning" for record in records)
    disagreement_count = sum(
        "rank03_branch_disagreement" in record["fusion"]["review_reasons"] for record in records
    )
    definitive_count = sum(
        record["fusion"]["visual_result"]
        in {
            VisualResult.MODEL_FLAGGED.value,
            VisualResult.MODEL_NOT_FLAGGED.value,
            VisualResult.NON_TB_ABNORMAL.value,
        }
        for record in records
    )
    escalation_count = sum(final_predictions)
    response_contract_failure_count = sum(
        not record["service"]["response_contract"]["contract_pass"] for record in records
    )
    return {
        "classifier_image_level": {
            **_binary_metrics_with_ci(
                classifier_labels,
                classifier_predictions,
                seed=seed + 101,
                replicates=bootstrap_replicates,
            ),
            "roc_auc": _roc_auc_with_ci(
                classifier_labels,
                classifier_scores,
                seed=seed + 111,
                replicates=bootstrap_replicates,
            ),
            "evaluation_scope": classifier_evaluation_scope,
            "excluded_technical_failures": technical_count,
            "excluded_argmax_ties": argmax_tie_count,
            "unevaluable_count": total - len(classifier_labels),
        },
        "detector_derived_image_level": {
            **_binary_metrics_with_ci(
                detector_labels,
                detector_predictions,
                seed=seed + 202,
                replicates=bootstrap_replicates,
            ),
            "roc_auc": _roc_auc_with_ci(
                detector_score_labels,
                detector_scores,
                seed=seed + 222,
                replicates=bootstrap_replicates,
            ),
            "status": detector_metric_status,
            "evaluation_scope": (
                "legacy image-level flag derived from maximum exported detector score; "
                "not an object-localization metric"
                if detector_records
                else "not applicable: detector output is advisory localization evidence only"
            ),
            "excluded_technical_failures": technical_count,
            "decision_not_applicable_count": advisory_detector_count,
            "score_evaluable_count": len(detector_score_records),
            "unevaluable_count": total - len(detector_labels),
        },
        "final_operational_triage": {
            **_binary_metrics_with_ci(
                final_labels,
                final_predictions,
                seed=seed + 303,
                replicates=bootstrap_replicates,
            ),
            "positive_action_definition": (
                "escalate=model_flagged, pending_human_review, indeterminate, or "
                "technical_failure; native non-TB classes remain direct classifier routes"
            ),
            "interpretation": "workflow disposition, not diagnosis",
        },
        "workflow_rates": {
            "visual_result_counts": dict(sorted(result_counts.items())),
            "successful_inference_rate": _rate(total - technical_count, total),
            "technical_failure_rate": _rate(technical_count, total),
            "classifier_argmax_tie_abstention_rate": _rate(argmax_tie_count, native_decision_count),
            "human_review_rate": _rate(review_count, total),
            "quality_warning_rate": _rate(quality_warning_count, total),
            "branch_disagreement_rate": _rate(disagreement_count, total),
            "definitive_output_coverage_rate": _rate(definitive_count, total),
            "escalation_rate": _rate(escalation_count, total),
            "response_contract_failure_rate": _rate(response_contract_failure_count, total),
        },
        "object_detection_localization": {
            "status": "not_computed_no_bbox_ground_truth",
            "bbox_ground_truth_available": False,
            "prohibited_metrics": ["AP", "AP50", "AP50:95", "IoU", "lesion recall"],
            "reason": (
                "The archive contains image-level labels only. Reporting localization AP, IoU, "
                "or lesion recall would be invalid."
            ),
        },
    }


def _write_record(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_ledger(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()


def run_shenzhen_evaluation(
    settings: Settings,
    *,
    archive_path: Path,
    output_root: Path | None = None,
    config_path: Path | None = None,
    _vision_backend: VisionBackend | None = None,
    _ledger_path: Path | None = None,
) -> Path:
    resolved_config = (
        config_path.resolve()
        if config_path is not None
        else (settings.project_root / "evaluation" / DEFAULT_CONFIG_NAME).resolve()
    )
    config = _validate_config(json.loads(resolved_config.read_text(encoding="utf-8")))
    policy_filename, expected_policy_id = _fusion_policy_binding(config)
    run_id = datetime.now(UTC).strftime("shenzhen-external-%Y%m%dT%H%M%S%fZ")
    run_root = (output_root or settings.data_root / DEFAULT_OUTPUT_DIRECTORY) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    records_path = run_root / "records.jsonl"
    records_path.touch()
    result_path = run_root / "result.json"
    isolated = replace(
        settings,
        data_root=run_root,
        db_path=run_root / "state.sqlite3",
        artifact_root=run_root / "artifacts",
        fusion_policy_filename=policy_filename,
        vision_backend="rank03",
        openai_enabled=False,
        narrator_backend="none",
        retain_uploaded_image=False,
    )

    started = time.perf_counter()
    runtime: dict[str, float] = {}
    archive_audit: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    service: TBXAgentService | None = None
    failure: Exception | None = None
    policy: dict[str, Any] | None = None
    policy_sha256: str | None = None
    expected_classifier_decision_rule: str | None = None
    expected_detector_decision_role: str | None = None
    runtime_config: dict[str, Any] | None = None
    sampler = _ProcessVramSampler(enabled=_vision_backend is None)
    torch_tracker = _TorchCudaTracker(enabled=_vision_backend is None)
    nvidia_measurement: dict[str, Any] = {
        "method": "not_started",
        "peak_bytes": 0,
    }
    torch_measurement: dict[str, Any] = {
        "method": "not_started",
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }
    try:
        policy_path = isolated.config_dir / policy_filename
        policy_sha256 = _sha256_file(policy_path)
        policy = isolated.fusion_policy()
        actual_policy_id = str(policy.get("policy_id", "")).strip()
        if actual_policy_id != expected_policy_id:
            raise ExternalEvaluationContractError(
                "fusion policy identity mismatch: "
                f"{actual_policy_id!r} != expected {expected_policy_id!r}"
            )
        (
            expected_classifier_decision_rule,
            expected_detector_decision_role,
        ) = _policy_decision_contract(policy)
        runtime_config = isolated.rank03_config()
        archive_audit, samples = _audit_archive(
            archive_path.resolve(),
            config=config,
            max_image_bytes=isolated.max_upload_bytes,
        )
        runtime["archive_hash_seconds"] = float(archive_audit["archive_hash_seconds"])
        runtime["archive_audit_seconds"] = float(archive_audit["archive_audit_seconds"])

        sampler.start()
        torch_tracker.start()
        initialization_started = time.perf_counter()
        service = TBXAgentService(isolated, vision_backend=_vision_backend)
        runtime["service_initialization_seconds"] = time.perf_counter() - initialization_started

        inference_started = time.perf_counter()
        payload_hashes: dict[str, str] = {}
        with (
            zipfile.ZipFile(archive_path.resolve(), "r") as archive,
            records_path.open("a", encoding="utf-8") as records_stream,
        ):
            for index, sample in enumerate(samples):
                item_started = time.perf_counter()
                payload_sha256: str | None = None
                try:
                    info = archive.getinfo(sample["entry_name"])
                    payload = _read_zip_entry_bounded(
                        archive,
                        info,
                        max_bytes=isolated.max_upload_bytes,
                    )
                    payload_sha256 = _sha256_bytes(payload)
                    case, response = service.assess_cxr(
                        payload,
                        user_id="shenzhen-external-evaluator",
                        owner_scope=f"external-eval:{run_id}:{index:04d}",
                        consent_to_process=True,
                        attested_chest_radiograph=True,
                    )
                    case, response = service.classify_cxr_case(
                        case_id=case.case_id,
                        owner_scope=case.owner_scope,
                        user_id="shenzhen-external-evaluator",
                        payload=payload,
                    )
                    record = _successful_record(
                        index=index,
                        sample=sample,
                        payload_sha256=payload_sha256,
                        case=case,
                        response=response,
                        expected_safety_policy_id=service.safety.policy_id,
                        expected_fusion_policy_id=expected_policy_id,
                        expected_classifier_decision_rule=expected_classifier_decision_rule,
                        expected_detector_decision_role=expected_detector_decision_role,
                        elapsed_seconds=time.perf_counter() - item_started,
                    )
                    duplicate_of = payload_hashes.get(payload_sha256)
                    record["duplicate_payload_of"] = duplicate_of
                    payload_hashes.setdefault(payload_sha256, sample["study_id"])
                except Exception as exc:
                    record = _exception_record(
                        index=index,
                        sample=sample,
                        payload_sha256=payload_sha256,
                        stage="zip_read_or_service_assessment",
                        exc=exc,
                        elapsed_seconds=time.perf_counter() - item_started,
                    )
                    record["duplicate_payload_of"] = None
                records.append(record)
                _write_record(records_stream, record)
        runtime["inference_seconds"] = time.perf_counter() - inference_started
    except Exception as exc:
        failure = exc
    finally:
        if service is not None:
            service.store.close()
        nvidia_measurement = sampler.stop()
        torch_measurement = torch_tracker.stop()

    metrics_started = time.perf_counter()
    confidence = config["confidence_intervals"]
    metrics = _calculate_metrics(
        records,
        seed=int(config["seed"]),
        bootstrap_replicates=int(confidence["replicates"]),
    )
    classifier_decision_evidence_count = sum(
        not record["technical_failure"]["occurred"]
        and (
            (
                record["classifier"].get("decision_rule") == "native_three_class_argmax"
                and (
                    record["classifier"].get("predicted_class") in {"healthy", "sick_non_tb", "tb"}
                    or record["classifier"].get("argmax_tied") is True
                )
            )
            or (
                record["classifier"].get("decision_rule") != "native_three_class_argmax"
                and record["classifier"].get("flagged") is not None
            )
        )
        for record in records
    )
    technical_failure_count = sum(record["technical_failure"]["occurred"] for record in records)
    if failure is None and classifier_decision_evidence_count == 0:
        failure = ExternalEvaluationContractError(
            "no image produced governed classifier decision evidence"
        )
    runtime["metrics_seconds"] = time.perf_counter() - metrics_started
    runtime["total_runtime_seconds"] = time.perf_counter() - started
    peak_vram_bytes = max(
        int(nvidia_measurement.get("peak_bytes", 0)),
        int(torch_measurement.get("peak_reserved_bytes", 0)),
    )
    records_sha256 = _sha256_file(records_path)
    source_revision = _source_revision(settings.project_root)
    source_tree_sha256 = _source_tree_sha256(settings.project_root)
    config_sha256 = _sha256_file(resolved_config)
    split_hash = (
        archive_audit["split_manifest_sha256"]
        if archive_audit is not None
        else config["dataset"]["expected_split_manifest_sha256"]
    )
    post_hoc = bool(
        config.get("prior_external_outcomes_observed") is True
        or str(config.get("analysis_status", "")).startswith("post_hoc")
    )
    if failure is not None:
        status = "failed_retained"
    elif technical_failure_count:
        status = (
            "completed_post_hoc_with_technical_failures_retained"
            if post_hoc
            else "completed_with_technical_failures_retained"
        )
    else:
        status = "completed_post_hoc_observational" if post_hoc else "completed_observational"
    result = {
        "run_id": run_id,
        "status": status,
        "hypothesis": config["hypothesis"],
        "full_configuration": {
            **config,
            "evaluation_config_path": _safe_relative(resolved_config, settings.project_root),
            "evaluation_config_sha256": config_sha256,
            "archive_audit": archive_audit,
            "service_runtime": {
                "vision_backend": "rank03",
                "test_backend_injected": _vision_backend is not None,
                "narrator_backend": "none",
                "openai_enabled": False,
                "retain_uploaded_image": False,
                "max_upload_bytes": isolated.max_upload_bytes,
                "fusion_policy_filename": policy_filename,
                "fusion_policy_sha256": policy_sha256,
                "expected_fusion_policy_id": expected_policy_id,
                "expected_classifier_decision_rule": expected_classifier_decision_rule,
                "expected_detector_decision_role": expected_detector_decision_role,
                "fusion_policy": policy,
                "rank03_runtime": runtime_config,
            },
            "environment": _environment_snapshot(),
        },
        "seed": int(config["seed"]),
        "seed_use": config["seed_use"],
        "split_hash": split_hash,
        "archive_sha256": (
            archive_audit["archive_sha256"]
            if archive_audit is not None
            else config["dataset"]["expected_archive_sha256"]
        ),
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "metrics": metrics,
        "record_count": len(records),
        "expected_record_count": int(config["dataset"]["expected_image_count"]),
        "records_path": str(records_path),
        "records_sha256": records_sha256,
        "database_path": str(isolated.db_path),
        "runtime": runtime,
        "runtime_seconds": runtime["total_runtime_seconds"],
        "peak_vram_bytes": peak_vram_bytes,
        "vram_measurement": {
            "nvidia_smi_process": nvidia_measurement,
            "torch_cuda_allocator": torch_measurement,
        },
        "locked_or_hidden_test_used": False,
        "clinical_validation": False,
        "failure": (
            {
                "error_type": type(failure).__name__,
                "error": str(failure)[:1000],
            }
            if failure is not None
            else None
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(result_path, result)

    ledger_entry = {
        key: value
        for key, value in result.items()
        if key not in {"created_at", "records_path", "database_path"}
    }
    ledger_entry["result_path"] = str(result_path)
    ledger_entry["records_path"] = str(records_path)
    ledger_path = _ledger_path or settings.project_root / "evaluation" / "ledger.jsonl"
    _append_ledger(ledger_path, ledger_entry)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "records": str(records_path),
                "status": status,
                "record_count": len(records),
            },
            ensure_ascii=False,
        )
    )
    if failure is not None:
        raise RuntimeError(f"external evaluation failed; retained at {result_path}") from failure
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen rank03 TBX-Agent assessment path on the complete, pinned Shenzhen "
            "external ZIP. No threshold, resize, subset, limit, or shuffle options are exposed."
        )
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run_shenzhen_evaluation(
        Settings.from_env(),
        archive_path=args.archive,
        output_root=args.output_root,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
