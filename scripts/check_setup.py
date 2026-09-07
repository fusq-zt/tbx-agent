#!/usr/bin/env python3
"""Read-only setup audit for TBX-Agent demo, vision, and RAG modes.

The command never downloads models, creates directories, writes configuration,
or probes a remote API. It only reports what is ready and what the operator must
prepare before starting the selected mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml

Status = Literal["PASS", "WARN", "FAIL"]
Mode = Literal["demo", "vision", "rag", "all"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_MARKERS = ("<", ">", "changeme", "placeholder", "replace-me")


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    message: str
    remediation: str | None = None


def _is_placeholder(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    lowered = value.strip().lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _load_dotenv(path: Path) -> tuple[dict[str, str], list[Check]]:
    """Parse the launcher's deliberately small dotenv grammar without mutation."""

    values: dict[str, str] = {}
    checks: list[Check] = []
    if not path.is_file():
        return values, [
            Check(
                "dotenv",
                "WARN",
                f"{path.name} is absent; process environment and defaults will be used",
                "Copy .env.example to .env and review it before a real run.",
            )
        ]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return values, [Check("dotenv", "FAIL", f"cannot read {path}: {type(exc).__name__}")]
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            checks.append(
                Check("dotenv", "FAIL", f"invalid .env line {line_number}: missing '='")
            )
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.replace("_", "a").isalnum() or key[0].isdigit():
            checks.append(Check("dotenv", "FAIL", f"invalid key on .env line {line_number}"))
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    checks.append(Check("dotenv", "PASS", f"parsed {path.name} without shell expansion"))
    return values, checks


def _effective_environment(dotenv: dict[str, str]) -> dict[str, str]:
    values = dict(dotenv)
    values.update(os.environ)
    return values


def _resolve_path(value: str | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if _is_placeholder(value):
        return None
    assert value is not None
    candidate = Path(value).expanduser()
    return (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (base / candidate).resolve()
    )


def _dependency_check(import_name: str, install_hint: str) -> Check:
    available = importlib.util.find_spec(import_name) is not None
    return Check(
        f"dependency:{import_name}",
        "PASS" if available else "FAIL",
        "installed" if available else "not importable",
        None if available else f"Install {install_hint} in the active Python environment.",
    )


def _file_check(name: str, path: Path, *, required: bool = True) -> Check:
    if path.is_file():
        return Check(name, "PASS", str(path))
    return Check(
        name,
        "FAIL" if required else "WARN",
        f"missing file: {path}",
        "Restore the file from the source checkout or point the related variable to a valid file.",
    )


def _directory_check(
    name: str,
    path: Path | None,
    *,
    required: bool,
    purpose: str,
) -> Check:
    if path is None:
        return Check(
            name,
            "FAIL" if required else "WARN",
            f"{purpose} path is not configured",
            "Set the documented environment variable to a repository-external directory.",
        )
    if path.is_dir():
        return Check(name, "PASS", str(path))
    if path.exists():
        return Check(name, "FAIL", f"expected a directory but found another path type: {path}")
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    return Check(
        name,
        "WARN" if writable and not required else "FAIL",
        f"directory does not exist: {path}",
        f"Create the {purpose} directory outside Git before starting the service.",
    )


def _check_python() -> Check:
    supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    return Check(
        "python",
        "PASS" if supported else "FAIL",
        f"{sys.version.split()[0]} ({sys.executable})",
        None if supported else "Use Python 3.11, 3.12, or 3.13.",
    )


def _check_rank03(env: dict[str, str]) -> list[Check]:
    configured = _resolve_path(env.get("TBX_AGENT_RANK03_RUNTIME_CONFIG"))
    if configured is None:
        return [
            Check(
                "vision:runtime-contract",
                "FAIL",
                "TBX_AGENT_RANK03_RUNTIME_CONFIG is not configured",
                "Download the published inference bundle, then run "
                "scripts/install_vision_bundle.py --bundle <zip> --artifact-root <path> "
                "--runtime-config <path> and set TBX_AGENT_RANK03_RUNTIME_CONFIG.",
            )
        ]
    check = _file_check("vision:runtime-contract", configured)
    if check.status == "FAIL":
        return [check]
    try:
        payload = json.loads(configured.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [Check("vision:runtime-contract", "FAIL", f"invalid JSON: {type(exc).__name__}")]
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    looks_like_template = "startup_instruction" in payload or "train and register" in serialized
    checks = [
        Check(
            "vision:runtime-contract",
            "FAIL" if looks_like_template else "PASS",
            "checked-in template is not an installed inference bundle contract"
            if looks_like_template
            else f"parsed external contract: {configured}",
            "Install the downloaded release bundle with scripts/install_vision_bundle.py "
            "--bundle <zip> --artifact-root <path> --runtime-config <path>."
            if looks_like_template
            else None,
        )
    ]
    model_root = _resolve_path(env.get("MODEL_WEIGHT_DIR") or env.get("TBX_ARTIFACT_ROOT"))
    checks.append(
        _directory_check(
            "vision:model-root",
            model_root,
            required=True,
            purpose="model weight",
        )
    )
    return checks


def _check_llm(env: dict[str, str]) -> list[Check]:
    provider = (
        env.get("LLM_PROVIDER")
        or env.get("TBX_AGENT_NARRATOR_BACKEND")
        or "none"
    ).strip().lower()
    if provider in {"", "none", "disabled"}:
        return [Check("llm", "PASS", "disabled; deterministic composition remains available")]
    if provider in {"openai", "openai_compatible"}:
        model = env.get("LLM_MODEL") or env.get("OPENAI_MODEL")
        base_url = env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL")
        key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
        return [
            Check(
                "llm:openai-compatible",
                "PASS" if model and base_url and key and not _is_placeholder(key) else "FAIL",
                "model/base URL/key configured (secret value not displayed)"
                if model and base_url and key and not _is_placeholder(key)
                else "model, base URL, or API key is missing",
                "Set LLM_* and matching OPENAI_* values in the local .env."
                if not (model and base_url and key and not _is_placeholder(key))
                else None,
            )
        ]
    if provider in {"llama_cpp", "qwen", "local"}:
        base_url = env.get("LLM_BASE_URL") or env.get("LLAMA_CPP_BASE_URL")
        model = env.get("LLM_MODEL") or env.get("LLAMA_CPP_MODEL_ALIAS")
        return [
            Check(
                "llm:local",
                "PASS" if base_url and model else "FAIL",
                "local endpoint/model configured: "
                f"{base_url or '<missing>'} / {model or '<missing>'}",
                None if base_url and model else "Configure LLAMA_CPP_BASE_URL and model alias.",
            )
        ]
    return [Check("llm", "FAIL", f"unsupported LLM_PROVIDER/narrator backend: {provider}")]


def _runtime_root_for_rag(env: dict[str, str]) -> Path:
    explicit = env.get("TBX_AGENT_DATA_ROOT") or env.get("TBX_RUNTIME_ROOT")
    configured = _resolve_path(explicit)
    if configured is not None:
        return configured
    if os.name == "nt":
        local_app_data = env.get("LOCALAPPDATA", "").strip()
        base = (
            Path(local_app_data).expanduser()
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return (base / "TBX-Agent" / "runtime").resolve()
    xdg_data = env.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg_data).expanduser() if xdg_data else Path.home() / ".local" / "share"
    return (base / "tbx-agent").resolve()


def _resolve_rag_vector_path(
    value: object,
    *,
    env: dict[str, str],
    config_path: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    prefix = "runtime://"
    if text.startswith(prefix):
        relative = PurePosixPath(text[len(prefix) :])
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            return None
        return _runtime_root_for_rag(env).joinpath(*relative.parts).resolve()
    return _resolve_path(text, base=config_path.parent)


def _check_rag(env: dict[str, str], *, selected_mode: Mode) -> list[Check]:
    checks: list[Check] = []
    config_path = _resolve_path(
        env.get("TBX_AGENT_RETRIEVAL_CONFIG")
        or str(PROJECT_ROOT / "configs" / "retrieval.yaml")
    )
    if config_path is None or not config_path.is_file():
        return [
            Check(
                "rag:config",
                "FAIL",
                f"retrieval config is missing: {config_path}",
                "Set TBX_AGENT_RETRIEVAL_CONFIG to an existing strict YAML contract.",
            )
        ]
    checks.append(Check("rag:config", "PASS", str(config_path)))
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        engine = raw["engine"]
        dense = raw["dense"]
        vector = raw["vector_store"]
        if not all(isinstance(item, dict) for item in (raw, engine, dense, vector)):
            raise TypeError("retrieval sections must be mappings")
        mode = str(engine["retrieval_mode"]).strip().lower()
    except (OSError, UnicodeError, KeyError, TypeError, yaml.YAMLError) as exc:
        return [
            *checks,
            Check(
                "rag:contract",
                "FAIL",
                f"retrieval YAML cannot be inspected: {type(exc).__name__}",
                "Validate it with the application preflight before startup.",
            ),
        ]
    if mode not in {"sparse", "dense", "hybrid"}:
        checks.append(Check("rag:mode", "FAIL", f"unsupported retrieval_mode={mode!r}"))
        return checks
    checks.append(Check("rag:mode", "PASS", mode))
    knowledge_dir = _resolve_path(
        env.get("TBX_AGENT_KNOWLEDGE_DIR") or str(PROJECT_ROOT / "knowledge")
    )
    if knowledge_dir is None:
        checks.append(Check("rag:knowledge", "FAIL", "runtime knowledge directory is unset"))
        return checks
    checks.extend(
        [
            _file_check("rag:source-manifest", knowledge_dir / "source_manifest.json"),
            _file_check("rag:chunks", knowledge_dir / "chunks.jsonl"),
        ]
    )
    source_dir = _resolve_path(env.get("GUIDELINE_SOURCE_DIR"))
    checks.append(
        _directory_check(
            "rag:raw-guidelines",
            source_dir,
            required=False,
            purpose="raw guideline source",
        )
    )
    if mode in {"dense", "hybrid"}:
        dense_enabled = dense.get("enabled") is True
        checks.append(
            Check(
                "rag:dense-enabled",
                "PASS" if dense_enabled else "FAIL",
                "enabled" if dense_enabled else "requested mode will fall back to sparse",
                None
                if dense_enabled
                else "Enable and hash-pin dense in the selected YAML contract.",
            )
        )
        adapter = str(dense.get("adapter", "")).strip()
        if adapter == "bge_m3_local":
            checks.append(_dependency_check("FlagEmbedding", "the retrieval extra"))
            model_path = _resolve_path(env.get("RAG_EMBEDDING_MODEL_PATH"))
            checks.append(
                _directory_check(
                    "rag:embedding-model",
                    model_path,
                    required=True,
                    purpose="offline BGE-M3 model",
                )
            )
            device = env.get("RAG_EMBEDDING_DEVICE", "").strip()
            checks.append(
                Check(
                    "rag:embedding-device",
                    "PASS" if device else "FAIL",
                    device or "RAG_EMBEDDING_DEVICE is unset",
                )
            )
        elif adapter != "openai_compatible_http":
            checks.append(
                Check(
                    "rag:dense-adapter",
                    "FAIL",
                    f"online adapter is unsupported: {adapter or '<missing>'}",
                )
            )
        model_sha = str(dense.get("model_sha256") or "")
        revision = str(dense.get("revision") or "").strip()
        pinned = len(model_sha) == 64 and all(char in "0123456789abcdef" for char in model_sha)
        checks.append(
            Check(
                "rag:embedding-provenance",
                "PASS" if pinned and revision else "FAIL",
                "model hash and revision pinned"
                if pinned and revision
                else "model_sha256 or revision is missing/invalid",
            )
        )
        checks.append(_dependency_check("qdrant_client", "the retrieval extra"))
        backend = str(vector.get("backend", "")).strip()
        checks.append(
            Check(
                "rag:vector-backend",
                "PASS" if backend == "qdrant_local" else "FAIL",
                backend or "not configured",
                None if backend == "qdrant_local" else "Online Dense/Hybrid requires qdrant_local.",
            )
        )
        vector_path = _resolve_rag_vector_path(
            vector.get("qdrant_path"),
            env=env,
            config_path=config_path,
        )
        checks.append(
            _directory_check(
                "rag:vector-index",
                vector_path,
                required=True,
                purpose="Qdrant Local vector index",
            )
        )
        if vector_path is not None and vector_path.is_dir():
            manifests = [
                vector_path / "manifest.json",
                vector_path / "index_manifest.json",
                vector_path / "index-manifest.json",
                vector_path / "current.json",
            ]
            if not any(path.is_file() for path in manifests):
                checks.append(
                    Check(
                        "rag:index-manifest",
                        "FAIL",
                        "vector directory has no recognized index manifest",
                        "Rebuild it with scripts/build_rag_index.py.",
                    )
                )
            else:
                checks.append(Check("rag:index-manifest", "PASS", "manifest found"))
    elif selected_mode in {"rag", "all"}:
        checks.append(Check("rag:vector-index", "PASS", "not required in sparse mode"))
    return checks


def run_checks(mode: Mode, dotenv_path: Path) -> list[Check]:
    dotenv, dotenv_checks = _load_dotenv(dotenv_path)
    env = _effective_environment(dotenv)
    if mode == "demo":
        # Match the launcher: a real .env must not make the model-free setup
        # check require language-model credentials or optional dense assets.
        env["LLM_PROVIDER"] = "none"
        env["TBX_AGENT_NARRATOR_BACKEND"] = "none"
        env["TBX_AGENT_RETRIEVAL_CONFIG"] = str(PROJECT_ROOT / "configs" / "retrieval.yaml")
    checks = [_check_python(), *dotenv_checks]
    config_paths = (
        "configs/app.yaml",
        "configs/retrieval.yaml",
        "configs/knowledge_ingestion.yaml",
    )
    for relative in config_paths:
        checks.append(_file_check(f"config:{Path(relative).name}", PROJECT_ROOT / relative))
    for dependency, hint in (
        ("fastapi", "the base package"),
        ("pydantic", "the base package"),
        ("yaml", "the base package"),
        ("uvicorn", "the base package"),
        ("streamlit", "the ui extra"),
        ("requests", "the ui extra"),
    ):
        checks.append(_dependency_check(dependency, hint))

    case_root = _resolve_path(
        env.get("CASE_DATA_DIR") or env.get("TBX_AGENT_DATA_ROOT")
    )
    checks.append(
        _directory_check(
            "runtime:case-data",
            case_root,
            required=mode in {"vision", "all"},
            purpose="case data",
        )
    )
    database_path = _resolve_path(env.get("TBX_AGENT_DB_PATH"))
    if database_path is None and case_root is not None:
        database_path = case_root / "tbx_agent.sqlite3"
    checks.append(
        _directory_check(
            "runtime:database-parent",
            database_path.parent if database_path is not None else None,
            required=mode in {"vision", "all"},
            purpose="database parent",
        )
    )
    if case_root is not None and case_root.resolve().is_relative_to(PROJECT_ROOT):
        checks.append(
            Check(
                "runtime:repository-boundary",
                "WARN",
                "case data is inside the Git checkout",
                "Use an external directory for real images and databases.",
            )
        )
    if mode in {"vision", "all"}:
        checks.extend(_check_rank03(env))
    if mode in {"rag", "all"}:
        checks.extend(_check_rag(env, selected_mode=mode))
    else:
        checks.extend(_check_rag(env, selected_mode=mode)[:4])
    checks.extend(_check_llm(env))
    if mode == "demo":
        checks.append(
            Check(
                "demo:backend",
                "PASS",
                "launch with -Demo/--demo to force the explicitly labelled mock backend",
            )
        )
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "vision", "rag", "all"), default="demo")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero for warnings as well as failures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checks = run_checks(args.mode, args.env_file.expanduser().resolve(strict=False))
    counts = {
        status: sum(check.status == status for check in checks)
        for status in ("PASS", "WARN", "FAIL")
    }
    result = {
        "mode": args.mode,
        "project_root": str(PROJECT_ROOT),
        "counts": counts,
        "checks": [asdict(check) for check in checks],
        "ready": counts["FAIL"] == 0 and (not args.strict or counts["WARN"] == 0),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"TBX-Agent setup check: mode={args.mode}")
        for check in checks:
            print(f"[{check.status:4}] {check.name}: {check.message}")
            if check.remediation:
                print(f"       -> {check.remediation}")
        print(f"Summary: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
