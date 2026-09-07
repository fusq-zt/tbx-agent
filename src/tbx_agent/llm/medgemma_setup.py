"""One-step, hash-verified MedGemma configuration for the local llama.cpp service.

The model file remains outside the repository and is used in place; this module
does not duplicate multi-gigabyte weights or create a model registry.  It only
verifies the requested Q4_K_M artifact and writes the external runtime pointer
files consumed by ``run_local``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT
from ..paths import default_runtime_root
from .bootstrap import (
    QwenBootstrapError,
    bootstrap_paths,
    ensure_api_key,
    install_windows_cuda_runtime,
    load_runtime_identity,
    load_source_contract,
)

MODEL_ALIAS = "tbx-medgemma-1.5-4b-it-q4-k-m"
MODEL_FILENAME = "medgemma-1.5-4b-it-Q4_K_M.gguf"
MODEL_SIZE_BYTES = 2_489_894_976
MODEL_SHA256 = "b31becdf4f39561800505514cce67681604fe449d04dd35c8c92fd7848c6d7bd"
MODEL_REPO_ID = "unsloth/medgemma-1.5-4b-it-GGUF"
MODEL_REVISION = "3855f948626b7ae42bccd082757f15078c53e758"
MODEL_ID = (
    f"{MODEL_REPO_ID}@{MODEL_REVISION}"
)


class MedGemmaSetupError(RuntimeError):
    """The selected MedGemma artifact or external runtime is unusable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        with contextlib.suppress(OSError):
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_identity(
    *,
    runtime_root: Path,
    environment: Mapping[str, str],
    install_windows_runtime: bool,
) -> tuple[Any, dict[str, str]]:
    paths = bootstrap_paths(runtime_root=runtime_root, environment=environment)
    if sys.platform.startswith("win") and install_windows_runtime:
        # The existing source contract contains the model-independent, pinned
        # llama.cpp b10517 Windows bundle. No Qwen model is downloaded here.
        contract = load_source_contract(PROJECT_ROOT / "configs/qwen_runtime_sources.yaml")
        identity = install_windows_cuda_runtime(contract, paths)
    else:
        try:
            identity = load_runtime_identity(paths)
        except Exception as exc:
            raise MedGemmaSetupError(
                "llama.cpp runtime is not installed; on Linux run "
                "scripts/build_llamacpp_linux.sh first, or on Windows omit "
                "--no-install-windows-runtime"
            ) from exc
    return paths, identity


def download_medgemma(
    *,
    runtime_root: Path,
    token: str | None = None,
) -> Path:
    """Download the immutable GGUF revision into the external runtime tree."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is part of the package.
        raise MedGemmaSetupError(
            "huggingface_hub is required for --download; reinstall TBX-Agent"
        ) from exc
    model_dir = runtime_root.expanduser().resolve(strict=False) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=model_dir,
            token=token or None,
        )
    except Exception as exc:
        raise MedGemmaSetupError(
            "MedGemma download failed. If Hugging Face requests access, accept the "
            "upstream terms and set HF_TOKEN before retrying."
        ) from exc
    return Path(downloaded).expanduser().resolve(strict=True)


def configure_medgemma(
    *,
    model_path: Path,
    runtime_root: Path,
    runtime_template: Path = PROJECT_ROOT / "configs/llm_runtime.yaml",
    environment: Mapping[str, str] | None = None,
    expected_sha256: str = MODEL_SHA256,
    expected_size_bytes: int = MODEL_SIZE_BYTES,
    install_windows_runtime: bool = True,
) -> dict[str, str]:
    """Verify one GGUF in place and write the external llama.cpp configuration."""

    environment = dict(os.environ if environment is None else environment)
    model = model_path.expanduser().resolve(strict=True)
    if not model.is_file():
        raise MedGemmaSetupError(f"MedGemma GGUF is not a regular file: {model}")
    if model.stat().st_size != expected_size_bytes:
        raise MedGemmaSetupError(
            "MedGemma GGUF size mismatch: expected "
            f"{expected_size_bytes}, observed {model.stat().st_size}"
        )
    actual_sha256 = _sha256_file(model)
    if actual_sha256 != expected_sha256:
        raise MedGemmaSetupError(
            "MedGemma GGUF SHA256 mismatch: expected "
            f"{expected_sha256}, observed {actual_sha256}"
        )

    runtime_root = runtime_root.expanduser().resolve(strict=False)
    paths, identity = _runtime_identity(
        runtime_root=runtime_root,
        environment=environment,
        install_windows_runtime=install_windows_runtime,
    )
    try:
        template = yaml.safe_load(runtime_template.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MedGemmaSetupError("MedGemma runtime template is unreadable") from exc
    if not isinstance(template, dict) or template.get("schema_version") != 1:
        raise MedGemmaSetupError("MedGemma runtime template schema is unsupported")

    api_key_file = ensure_api_key(paths.api_key_file).resolve()
    template.update(
        {
            "runtime_id": "medgemma15-4b-it-q4-k-m-llamacpp-b10517-v1",
            "release_archive_sha256": identity["release_archive_sha256"],
            "binary_path": identity["binary_path"],
            "binary_sha256": identity["binary_sha256"],
            "bundle_manifest_path": identity["bundle_manifest_path"],
            "bundle_manifest_sha256": identity["bundle_manifest_sha256"],
            "api_key_file": str(api_key_file),
            "model_id": MODEL_ID,
            "model_alias": MODEL_ALIAS,
            "model_path": str(model),
            "model_sha256": expected_sha256,
            "quantization": "Q4_K_M",
            "load_mmproj": False,
        }
    )
    encoded = yaml.safe_dump(template, allow_unicode=True, sort_keys=False).encode("utf-8")
    _write_atomic(paths.runtime_config, encoded)

    env_values = {
        "TBX_AGENT_LLM_RUNTIME_CONFIG": str(paths.runtime_config.resolve()),
        "LLAMA_CPP_BINARY_PATH": identity["binary_path"],
        "LLAMA_CPP_BUNDLE_MANIFEST_PATH": identity["bundle_manifest_path"],
        "LLAMA_CPP_API_KEY_FILE": str(api_key_file),
        "LLAMA_CPP_MODEL_ALIAS": MODEL_ALIAS,
        "LLAMA_CPP_MODEL_PATH": str(model),
        "LLAMA_CPP_MODEL_SHA256": expected_sha256,
    }
    if any("\n" in value or "\r" in value for value in env_values.values()):
        raise MedGemmaSetupError("runtime path contains an unsupported newline")
    _write_atomic(
        paths.environment_file,
        "".join(f"{key}={value}\n" for key, value in env_values.items()).encode("utf-8"),
    )
    return {
        "model": str(model),
        "model_alias": MODEL_ALIAS,
        "model_sha256": actual_sha256,
        "runtime_config": str(paths.runtime_config.resolve()),
        "environment_file": str(paths.environment_file.resolve()),
        "runtime_id": str(template["runtime_id"]),
        "mmproj_loaded": "false",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify MedGemma 1.5 4B Q4_K_M and configure the external llama.cpp runtime"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="already downloaded Q4_K_M GGUF")
    source.add_argument(
        "--download",
        action="store_true",
        help="download the pinned GGUF revision to <runtime-root>/models",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="external runtime root (default: TBX_RUNTIME_ROOT/platform runtime root)",
    )
    parser.add_argument(
        "--runtime-template",
        type=Path,
        default=PROJECT_ROOT / "configs/llm_runtime.yaml",
    )
    parser.add_argument(
        "--no-install-windows-runtime",
        action="store_true",
        help="require an already registered llama.cpp runtime",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime_root = args.runtime_root or default_runtime_root()
    try:
        model_path = (
            download_medgemma(
                runtime_root=runtime_root,
                token=os.getenv("HF_TOKEN", "").strip() or None,
            )
            if args.download
            else args.model
        )
        assert model_path is not None
        result = configure_medgemma(
            model_path=model_path,
            runtime_root=runtime_root,
            runtime_template=args.runtime_template,
            install_windows_runtime=not args.no_install_windows_runtime,
        )
    except (OSError, ValueError, QwenBootstrapError, MedGemmaSetupError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2))
    return 0
