"""Install a bounded, hash-checked inference bundle without deserializing weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

PAYLOAD_FILES = frozenset({
    "classifier.pt", "detector.pt", "classifier_config.json", "detector_config.json",
})
BUNDLE_FILES = PAYLOAD_FILES | {"manifest.json"}
MAX_JSON_BYTES = 1024 * 1024
MAX_WEIGHT_BYTES = 2 * 1024**3
MAX_BUNDLE_BYTES = 4 * 1024**3
MAX_COMPRESSION_RATIO = 100
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DETECTOR_KEYS = {
    "DFINE", "DFINEPostProcessor", "DFINETransformer", "HGNetv2", "HybridEncoder",
    "eval_spatial_size", "model", "num_classes", "postprocessor", "remap_mscoco_category",
    "use_focal_loss",
}
_CLASSIFIER_CONTRACT = {
    "architecture": "convnext_tiny.fb_in1k", "input_size": 320,
    "probability_order": ["healthy", "sick_non_tb", "tb"],
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225], "amp": True,
}
_DETECTOR_CONTRACT = {
    "architecture": "D-FINE-L HGNetv2-B4", "input_size": 512, "native_label": 0,
    "native_top_queries": 300, "export_floor": 0.05,
    "bbox_format": "xyxy_original_image_pixels", "tta": "none", "precision": "fp32",
}


class BundleError(ValueError):
    """An inference bundle is invalid, unsafe, or cannot be installed exclusively."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(raw: bytes, label: str) -> dict[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        raise BundleError(f"{label} exceeds the JSON size limit")
    def reject_constant(value: str) -> None:
        raise BundleError(f"non-finite JSON value: {value}")

    try:
        result = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BundleError(f"invalid {label}") from exc
    if not isinstance(result, dict):
        raise BundleError(f"{label} must contain a JSON object")
    return result


def _same_json(left: Any, right: Any) -> bool:
    # JSON's distinct bool and integer types must not collapse through True == 1.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _check_contract(observed: Any, required: dict[str, Any], label: str) -> None:
    if not isinstance(observed, dict):
        raise BundleError(f"{label} must be an object")
    for key, expected in required.items():
        if key not in observed or not _same_json(observed[key], expected):
            raise BundleError(f"unsupported {label}.{key}")


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the public inference-only bundle schema; paths are never trusted."""

    if set(value) != {"schema_version", "bundle_type", "bundle_id", "files", "runtime"}:
        raise BundleError("manifest fields must match the inference bundle schema exactly")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise BundleError("unsupported bundle schema_version")
    if value["bundle_type"] != "rank03-inference-v1":
        raise BundleError("unsupported bundle_type")
    bundle_id = value["bundle_id"]
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
    }
    if (
        not isinstance(bundle_id, str) or not _BUNDLE_ID.fullmatch(bundle_id)
        or bundle_id.endswith(".") or bundle_id.split(".")[0] in reserved
    ):
        raise BundleError("bundle_id must be a safe lowercase label")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != PAYLOAD_FILES:
        raise BundleError("manifest must describe exactly the four inference payload files")
    for name, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}:
            raise BundleError(f"invalid manifest file record: {name}")
        if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(entry["sha256"]):
            raise BundleError(f"invalid SHA-256 for {name}")
        maximum = MAX_JSON_BYTES if name.endswith(".json") else MAX_WEIGHT_BYTES
        size = entry["size_bytes"]
        if type(size) is not int or not 0 < size <= maximum:
            raise BundleError(f"invalid or excessive file size for {name}")
    if sum(entry["size_bytes"] for entry in files.values()) > MAX_BUNDLE_BYTES:
        raise BundleError("bundle exceeds the total size limit")
    runtime = value["runtime"]
    runtime_keys = {
        "schema_version", "template", "model_bundle_id", "source_revision", "classifier",
        "detector", "validation_scope", "weights_distributed_by_repository", "startup_instruction",
    }
    if not isinstance(runtime, dict) or set(runtime) - runtime_keys:
        raise BundleError("runtime contains unsupported fields")
    if type(runtime.get("schema_version")) is not int or runtime["schema_version"] != 1:
        raise BundleError("unsupported runtime schema_version")
    classifier, detector = runtime.get("classifier"), runtime.get("detector")
    _check_contract(classifier, _CLASSIFIER_CONTRACT, "runtime.classifier")
    _check_contract(detector, _DETECTOR_CONTRACT, "runtime.detector")
    if set(classifier) - set(_CLASSIFIER_CONTRACT) - {
        "checkpoint_path", "checkpoint_sha256", "config_path", "config_sha256",
    }:
        raise BundleError("runtime classifier contains unsupported fields")
    if set(detector) - set(_DETECTOR_CONTRACT) - {
        "checkpoint_path", "checkpoint_sha256", "resolved_config_path", "resolved_config_sha256",
        "source_root", "base_config_path",
    }:
        raise BundleError("runtime detector contains unsupported fields")
    return value


def _validate_configs(classifier: dict[str, Any], detector: dict[str, Any]) -> None:
    if set(classifier) != {"model", "input_size", "amp"}:
        raise BundleError("classifier config must contain only model, input_size, amp")
    _check_contract(classifier, {
        "model": "convnext_tiny.fb_in1k", "input_size": 320, "amp": True,
    }, "classifier config")
    if set(detector) != _DETECTOR_KEYS:
        raise BundleError("detector config must contain only the inference architecture fields")
    _check_contract(detector, {
        "model": "DFINE", "postprocessor": "DFINEPostProcessor", "num_classes": 1,
        "remap_mscoco_category": False, "eval_spatial_size": [512, 512], "use_focal_loss": True,
        "DFINE": {"backbone": "HGNetv2", "encoder": "HybridEncoder", "decoder": "DFINETransformer"},
    }, "detector config")
    _check_contract(detector["DFINEPostProcessor"], {"num_top_queries": 300}, "postprocessor")
    _check_contract(detector["HGNetv2"], {
        "name": "B4", "pretrained": False, "return_idx": [1, 2, 3],
    }, "HGNetv2")
    _check_contract(detector["HybridEncoder"], {
        "hidden_dim": 256, "in_channels": [512, 1024, 2048], "feat_strides": [8, 16, 32],
        "nhead": 8, "num_encoder_layers": 1,
    }, "HybridEncoder")
    _check_contract(detector["DFINETransformer"], {
        "hidden_dim": 256, "num_layers": 6, "num_levels": 3, "num_queries": 300,
        "feat_channels": [256, 256, 256], "feat_strides": [8, 16, 32], "reg_max": 32,
    }, "DFINETransformer")

    def check_nested(item: Any, depth: int = 0) -> None:
        if depth > 16:
            raise BundleError("detector config exceeds the nesting limit")
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "pretrained" and child is False:
                    continue
                if re.search(
                    r"path|dir|file|checkpoint|pretrain|tuning|resume|include|train|dataset|"
                    r"optimizer|scheduler|criterion", key, re.IGNORECASE,
                ):
                    raise BundleError(f"detector config contains a non-inference setting: {key}")
                check_nested(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                check_nested(child, depth + 1)
        elif isinstance(item, str) and not re.fullmatch(r"[A-Za-z0-9_+-]{1,64}", item):
            raise BundleError("detector config contains a path or unsupported string setting")

    check_nested(detector)


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _safe_path(path: Path) -> Path:
    absolute = path.expanduser().absolute()
    for ancestor in (*reversed(absolute.parents), absolute):
        if os.path.lexists(ancestor) and _is_link(ancestor):
            raise BundleError(f"symlinks and reparse points are prohibited: {ancestor}")
    return absolute.resolve(strict=False)


def _external_path(path: Path) -> Path:
    resolved = _safe_path(path)
    if resolved.is_relative_to(_PROJECT_ROOT):
        raise BundleError("artifacts and generated runtime config must be outside the project")
    return resolved


def _mkdir_safe(path: Path) -> None:
    _safe_path(path)
    path.mkdir(parents=True, exist_ok=True)
    _safe_path(path)


@contextmanager
def _open_regular(path: Path) -> Iterator[BinaryIO]:
    _safe_path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise BundleError("bundle input must be a regular file")
        _safe_path(path)
        yield stream


def _archive_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) != len(BUNDLE_FILES) or {item.filename for item in infos} != BUNDLE_FILES:
        raise BundleError("ZIP must contain exactly five allowlisted files at its root")
    for item in infos:
        file_type = stat.S_IFMT(item.external_attr >> 16)
        if (
            item.is_dir() or file_type not in {0, stat.S_IFREG} or item.flag_bits & 1
            or item.external_attr & (0x400 | 0x10)
        ):
            raise BundleError("ZIP links, directories, special files and encryption are prohibited")
        if item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise BundleError("unsupported ZIP compression method")
        maximum = MAX_JSON_BYTES if item.filename.endswith(".json") else MAX_WEIGHT_BYTES
        if not 0 < item.file_size <= maximum:
            raise BundleError("ZIP member exceeds its size limit")
        if item.file_size > max(1, item.compress_size) * MAX_COMPRESSION_RATIO:
            raise BundleError("ZIP compression ratio exceeds the limit")
    if sum(item.file_size for item in infos) > MAX_BUNDLE_BYTES:
        raise BundleError("ZIP exceeds the total expanded size limit")
    return {item.filename: item for item in infos}


def _copy_verified(source: BinaryIO, destination: Path, entry: dict[str, Any]) -> None:
    digest, total = hashlib.sha256(), 0
    with destination.open("xb") as output:
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > entry["size_bytes"]:
                raise BundleError(f"file exceeds declared size: {destination.name}")
            digest.update(block)
            output.write(block)
    if total != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
        raise BundleError(f"size or SHA-256 mismatch: {destination.name}")


def _snapshot_archive(path: Path, snapshot: Path, expected: str | None) -> str:
    # Hash exactly the immutable private copy subsequently parsed, avoiding a
    # pathname replacement between archive verification and extraction.
    if expected is not None and not _SHA256.fullmatch(expected):
        raise BundleError("--sha256 must be a lowercase SHA-256 digest")
    digest, total = hashlib.sha256(), 0
    with _open_regular(path) as stream, snapshot.open("xb") as output:
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > MAX_BUNDLE_BYTES:
                raise BundleError("archive exceeds the size limit")
            digest.update(block)
            output.write(block)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise BundleError("bundle archive SHA-256 mismatch")
    return actual


def _runtime_config(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = json.loads(json.dumps(manifest["runtime"]))
    prefix = f"artifact://rank03/bundles/{manifest['bundle_id']}"
    files = manifest["files"]
    runtime.update(
        template=False, model_bundle_id=manifest["bundle_id"],
        weights_distributed_by_repository=False,
        startup_instruction="Set TBX_ARTIFACT_ROOT and TBX_AGENT_RANK03_RUNTIME_CONFIG.",
    )
    runtime["classifier"].update(
        checkpoint_path=f"{prefix}/classifier.pt",
        checkpoint_sha256=files["classifier.pt"]["sha256"],
        config_path=f"{prefix}/classifier_config.json",
        config_sha256=files["classifier_config.json"]["sha256"],
    )
    runtime["detector"].update(
        checkpoint_path=f"{prefix}/detector.pt",
        checkpoint_sha256=files["detector.pt"]["sha256"],
        resolved_config_path=f"{prefix}/detector_config.json",
        resolved_config_sha256=files["detector_config.json"]["sha256"],
        source_root="artifact://sources/D-FINE",
        base_config_path="artifact://sources/D-FINE/configs/dfine/dfine_hgnetv2_l_coco.yml",
    )
    return runtime


def install_rank03_bundle(
    bundle: str | Path, *, artifact_root: str | Path, runtime_config: str | Path,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Verify and exclusively install a local directory or ZIP. Never import torch."""

    source = _safe_path(Path(bundle))
    root, config_path = _external_path(Path(artifact_root)), _external_path(Path(runtime_config))
    if os.path.lexists(config_path):
        raise BundleError("runtime config already exists; choose a new output path")
    if not source.is_dir() and not source.is_file():
        raise BundleError("bundle must be a local directory or ZIP file")
    archive_sha256: str | None = None
    if source.is_dir():
        if sha256 is not None:
            raise BundleError("--sha256 applies to ZIP archives, not directories")
        if root.is_relative_to(source) or config_path.is_relative_to(source):
            raise BundleError("installation outputs must not modify the input bundle directory")
        children = tuple(source.iterdir())
        if {child.name for child in children} != BUNDLE_FILES:
            raise BundleError("bundle directory must contain exactly the five allowlisted files")
        for child in children:
            _safe_path(child)
            if not child.is_file():
                raise BundleError("bundle directory may contain only regular files")
    elif sha256 is not None and not _SHA256.fullmatch(sha256):
        raise BundleError("--sha256 must be a lowercase SHA-256 digest")
    _mkdir_safe(root)
    with tempfile.TemporaryDirectory(prefix=".rank03-install-", dir=root) as staging_name:
        staging = Path(staging_name)
        if source.is_dir():
            with _open_regular(source / "manifest.json") as stream:
                manifest_raw = stream.read(MAX_JSON_BYTES + 1)
            manifest = validate_manifest(_read_json(manifest_raw, "manifest.json"))
            for name in sorted(PAYLOAD_FILES):
                with _open_regular(source / name) as stream:
                    _copy_verified(stream, staging / name, manifest["files"][name])
        else:
            snapshot = staging / ".bundle.zip"
            archive_sha256 = _snapshot_archive(source, snapshot, sha256)
            try:
                with (
                    _open_regular(snapshot) as archive_stream,
                    zipfile.ZipFile(archive_stream) as archive,
                ):
                    entries = _archive_entries(archive)
                    with archive.open("manifest.json") as stream:
                        manifest_raw = stream.read(MAX_JSON_BYTES + 1)
                    manifest = validate_manifest(_read_json(manifest_raw, "manifest.json"))
                    for name in sorted(PAYLOAD_FILES):
                        if entries[name].file_size != manifest["files"][name]["size_bytes"]:
                            raise BundleError(f"ZIP size does not match manifest: {name}")
                        with archive.open(name) as stream:
                            _copy_verified(stream, staging / name, manifest["files"][name])
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
                raise BundleError("invalid or unsupported bundle ZIP") from exc
        _validate_configs(
            _read_json((staging / "classifier_config.json").read_bytes(), "classifier config"),
            _read_json((staging / "detector_config.json").read_bytes(), "detector config"),
        )
        (staging / "manifest.json").write_bytes(manifest_raw)
        installed = root / "rank03" / "bundles" / manifest["bundle_id"]
        if config_path.is_relative_to(installed):
            raise BundleError("runtime config must not be inside the installed bundle")
        _mkdir_safe(installed.parent)
        _mkdir_safe(config_path.parent)
        _safe_path(installed)
        # mkdir and exclusive file creation provide no-overwrite behavior even
        # when two installer processes target the same ID or config output.
        try:
            installed.mkdir()
        except FileExistsError as exc:
            raise BundleError("bundle already installed; choose a new bundle ID") from exc
        config_created = False
        try:
            for name in sorted(BUNDLE_FILES):
                os.rename(staging / name, installed / name)
            _safe_path(config_path)
            with config_path.open("xb") as output:
                config_created = True
                output.write(_json_bytes(_runtime_config(manifest)))
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            if config_created:
                config_path.unlink(missing_ok=True)
            if not _safe_path(installed).is_relative_to(root):
                raise BundleError("refusing cleanup outside the artifact root") from None
            shutil.rmtree(installed)
            raise
    return {
        "status": "installed", "bundle_id": manifest["bundle_id"],
        "artifact_root": str(root), "installed_directory": str(installed),
        "runtime_config": str(config_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "bundle_sha256": archive_sha256, "archive_pin_verified": sha256 is not None,
        "weights_deserialized": False,
        "environment": {
            "TBX_ARTIFACT_ROOT": str(root), "TBX_AGENT_RANK03_RUNTIME_CONFIG": str(config_path),
        },
    }


def _https_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise BundleError("invalid HTTPS bundle URL") from exc
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
        or parsed.fragment
    ):
        raise BundleError("bundle downloads and redirects require credential-free HTTPS URLs")
    return url


class _HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _https_url(newurl))


def download_and_install_rank03_bundle(
    url: str, *, sha256: str, artifact_root: str | Path, runtime_config: str | Path,
) -> dict[str, Any]:
    """Download a trusted-hash HTTPS ZIP with bounded redirects, bytes and timeout."""

    _https_url(url)
    if not _SHA256.fullmatch(sha256):
        raise BundleError("HTTPS downloads require a trusted lowercase --sha256 digest")
    root = _external_path(Path(artifact_root))
    config_path = _external_path(Path(runtime_config))
    if os.path.lexists(config_path):
        raise BundleError("runtime config already exists; choose a new output path")
    _mkdir_safe(root)
    with tempfile.TemporaryDirectory(prefix=".rank03-download-", dir=root) as temporary:
        destination = Path(temporary) / "bundle.zip"
        opener = urllib.request.build_opener(_HTTPSRedirectHandler())
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "TBX-Agent/1"})
            with opener.open(request, timeout=30) as response, destination.open("xb") as output:
                _https_url(response.geturl())
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_BUNDLE_BYTES:
                        raise BundleError("bundle download exceeds the size limit")
                    output.write(block)
        except (OSError, urllib.error.URLError) as exc:
            # Signed URLs and service credentials are not copied into diagnostics.
            raise BundleError("HTTPS bundle download failed") from exc
        return install_rank03_bundle(
            destination, artifact_root=root, runtime_config=config_path, sha256=sha256,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="local directory or ZIP with five root files")
    source.add_argument("--url", help="HTTPS bundle ZIP; requires a trusted --sha256")
    parser.add_argument("--sha256", help="trusted ZIP SHA-256 (required for --url)")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.url and not args.sha256:
        parser.error("--url requires a trusted --sha256")
    try:
        if args.url:
            result = download_and_install_rank03_bundle(
                args.url, sha256=args.sha256, artifact_root=args.artifact_root,
                runtime_config=args.runtime_config,
            )
        else:
            result = install_rank03_bundle(
                args.bundle, artifact_root=args.artifact_root, runtime_config=args.runtime_config,
                sha256=args.sha256,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (BundleError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
