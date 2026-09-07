"""Command line interface for the external Qwen3.5/llama.cpp runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from .bootstrap import (
    QwenBootstrapError,
    bootstrap_paths,
    build_model,
    configure_runtime,
    detect_quantize_binary,
    download_qwen_source,
    install_llama_source,
    install_windows_cuda_runtime,
    load_runtime_identity,
    load_source_contract,
    public_status,
    register_model,
    register_runtime_bundle,
    verify_prepared_runtime,
)


def _default_project_root() -> Path:
    return PROJECT_ROOT


def _parser() -> argparse.ArgumentParser:
    project = _default_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Build or register the exact Qwen3.5-4B Q4_K_M artifact and pinned "
            "llama.cpp runtime outside the source repository."
        )
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=project / "configs/qwen_runtime_sources.yaml",
    )
    parser.add_argument(
        "--runtime-template", type=Path, default=project / "configs/llm_runtime.yaml"
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("status", help="show secret-free external artifact status")
    subparsers.add_parser(
        "install-runtime",
        help="install the pinned Windows CUDA b10517 release (Windows x64 only)",
    )
    subparsers.add_parser(
        "download-source", help="download and verify the pinned official Qwen snapshot"
    )
    subparsers.add_parser(
        "download-llama-source",
        help="download and verify the pinned b10517 source archive for a local build",
    )

    register_model_parser = subparsers.add_parser(
        "register-model", help="copy an existing exact Q4_K_M GGUF into the artifact cache"
    )
    register_model_parser.add_argument("model", type=Path)
    register_model_parser.add_argument("--configure", action="store_true")

    build_parser = subparsers.add_parser(
        "build-model", help="download official sources, convert, quantize, and verify"
    )
    build_parser.add_argument("--quantize-binary", type=Path)
    build_parser.add_argument("--keep-bf16", action="store_true")
    build_parser.add_argument("--configure", action="store_true")

    register_runtime_parser = subparsers.add_parser(
        "register-runtime",
        help="register a user-built b10517 bundle for a non-Windows or custom target",
    )
    register_runtime_parser.add_argument("--bundle-dir", type=Path, required=True)
    register_runtime_parser.add_argument("--binary-name", required=True)
    register_runtime_parser.add_argument("--expected-binary-sha256", required=True)
    register_runtime_parser.add_argument("--release-archive", type=Path, required=True)
    register_runtime_parser.add_argument("--expected-release-sha256", required=True)
    register_runtime_parser.add_argument("--platform-id", required=True)

    subparsers.add_parser(
        "configure", help="write an external runtime YAML, API key file, and secret-free env file"
    )
    subparsers.add_parser("verify", help="verify all configured runtime files by digest")

    prepare = subparsers.add_parser(
        "prepare",
        help="one-shot Windows runtime install plus exact model registration or official build",
    )
    choice = prepare.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", type=Path, help="existing exact Q4_K_M GGUF")
    choice.add_argument("--build-from-official", action="store_true")
    prepare.add_argument("--keep-bf16", action="store_true")
    return parser


def _configure(
    *,
    contract: dict[str, Any],
    paths,
    template: Path,
) -> Path:
    return configure_runtime(
        contract,
        paths,
        runtime_identity=load_runtime_identity(paths),
        template_path=template.resolve(strict=True),
    )


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = load_source_contract(args.sources.resolve(strict=True))
        paths = bootstrap_paths(
            artifact_root=args.artifact_root,
            runtime_root=args.runtime_root,
        )
        payload: dict[str, object] = {"status": "ok", "action": args.action}

        if args.action == "status":
            payload.update(public_status(paths))
        elif args.action == "install-runtime":
            if not sys.platform.startswith("win"):
                raise QwenBootstrapError(
                    "automatic runtime installation is pinned only for Windows CUDA 13.3 x64; "
                    "build b10517 for this platform and use register-runtime"
                )
            identity = install_windows_cuda_runtime(contract, paths)
            payload.update({"runtime_id": identity["runtime_id"]})
        elif args.action == "download-source":
            download_qwen_source(contract, paths)
            payload.update({"source_verified": True})
        elif args.action == "download-llama-source":
            source_path = install_llama_source(contract, paths)
            payload.update(
                {
                    "source_verified": True,
                    "source_path": str(source_path.resolve()),
                    "source_archive_sha256": contract["llama_cpp"]["source_archive"][
                        "sha256"
                    ],
                }
            )
        elif args.action == "register-model":
            register_model(contract, paths, args.model)
            if args.configure:
                _configure(contract=contract, paths=paths, template=args.runtime_template)
            payload.update({"model_verified": True, "configured": bool(args.configure)})
        elif args.action == "build-model":
            identity = load_runtime_identity(paths)
            quantize = args.quantize_binary or detect_quantize_binary(identity)
            build_model(
                contract,
                paths,
                quantize_binary=quantize,
                keep_bf16=args.keep_bf16,
            )
            if args.configure:
                _configure(contract=contract, paths=paths, template=args.runtime_template)
            payload.update({"model_verified": True, "configured": bool(args.configure)})
        elif args.action == "register-runtime":
            identity = register_runtime_bundle(
                contract,
                paths,
                bundle_dir=args.bundle_dir,
                binary_name=args.binary_name,
                expected_binary_sha256=args.expected_binary_sha256,
                release_archive=args.release_archive,
                expected_release_sha256=args.expected_release_sha256,
                platform_id=args.platform_id,
            )
            payload.update({"runtime_id": identity["runtime_id"]})
        elif args.action == "configure":
            _configure(contract=contract, paths=paths, template=args.runtime_template)
            payload.update({"configured": True})
        elif args.action == "verify":
            payload["attestation"] = verify_prepared_runtime(paths.runtime_config)
        elif args.action == "prepare":
            identity = (
                install_windows_cuda_runtime(contract, paths)
                if sys.platform.startswith("win")
                else load_runtime_identity(paths)
            )
            if args.model is not None:
                register_model(contract, paths, args.model)
            else:
                build_model(
                    contract,
                    paths,
                    quantize_binary=detect_quantize_binary(identity),
                    keep_bf16=args.keep_bf16,
                )
            _configure(contract=contract, paths=paths, template=args.runtime_template)
            payload.update(
                {
                    "runtime_id": identity["runtime_id"],
                    "model_verified": True,
                    "configured": True,
                }
            )
        _print(payload)
        return 0
    except (OSError, QwenBootstrapError, ValueError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
