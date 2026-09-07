from __future__ import annotations

import json
import ntpath
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from .artifacts import default_artifact_root
from .paths import default_runtime_root, discover_project_root

PROJECT_ROOT = discover_project_root()
DEFAULT_FUSION_POLICY_FILENAME = "fusion_policy.json"


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_data_root() -> Path:
    return default_runtime_root()


def _resolve_config_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve(strict=False)
        if path.is_absolute()
        else (base / path).resolve(strict=False)
    )


def local_paths_equivalent(left: str | Path, right: str | Path) -> bool:
    """Compare local path identities without treating Windows case as drift.

    Runtime contracts remain fail-closed: only lexical Windows case/separator
    differences and native aliases resolved by the operating system compare equal.
    Different filenames, drives, or unresolved locations remain different.
    """

    left_text = str(left).strip()
    right_text = str(right).strip()
    left_windows = PureWindowsPath(left_text)
    right_windows = PureWindowsPath(right_text)
    if left_windows.is_absolute() or right_windows.is_absolute():
        if not (left_windows.is_absolute() and right_windows.is_absolute()):
            return False
        return ntpath.normcase(ntpath.normpath(str(left_windows))) == ntpath.normcase(
            ntpath.normpath(str(right_windows))
        )

    left_path = Path(left_text).expanduser().resolve(strict=False)
    right_path = Path(right_text).expanduser().resolve(strict=False)
    try:
        if left_path.exists() and right_path.exists():
            return left_path.samefile(right_path)
    except OSError:
        pass
    return os.path.normcase(str(left_path)) == os.path.normcase(str(right_path))


def local_paths_overlap(left: str | Path, right: str | Path) -> bool:
    """Return true when two resolved roots are equal or contain one another."""

    left_path = Path(left).expanduser().resolve(strict=False)
    right_path = Path(right).expanduser().resolve(strict=False)
    return (
        local_paths_equivalent(left_path, right_path)
        or left_path.is_relative_to(right_path)
        or right_path.is_relative_to(left_path)
    )


def _load_llama_cpp_api_key(
    *,
    narrator_backend: str,
    configured_file: str,
    project_root: Path,
) -> str:
    direct = os.getenv("LLAMA_CPP_API_KEY", "").strip()
    if direct or narrator_backend != "llama_cpp":
        return direct
    file_value = os.getenv("LLAMA_CPP_API_KEY_FILE", configured_file).strip()
    if not file_value:
        return ""
    secret_path = _resolve_config_path(file_value, base=project_root)
    try:
        keys = [
            line.strip()
            for line in secret_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as exc:
        raise ValueError("configured llama.cpp API key file is unreadable") from exc
    if len(keys) != 1 or len(keys[0]) < 32:
        raise ValueError("llama.cpp API key file must contain exactly one strong key")
    return keys[0]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    config_dir: Path
    knowledge_dir: Path
    retrieval_config_path: Path
    data_root: Path
    db_path: Path
    artifact_root: Path
    fusion_policy_filename: str
    vision_backend: str
    openai_enabled: bool
    openai_model: str
    narrator_backend: str
    ollama_base_url: str
    ollama_model: str
    ollama_model_digest: str
    ollama_timeout_seconds: float
    ollama_max_response_bytes: int
    ollama_allow_remote: bool
    max_upload_bytes: int
    retain_uploaded_image: bool
    require_real_inference: bool = False
    require_llm_inference: bool = False
    llama_cpp_base_url: str = "http://127.0.0.1:11435"
    llama_cpp_model_alias: str = "tbx-medgemma-1.5-4b-it-q4-k-m"
    llama_cpp_model_path: str = ""
    llama_cpp_model_sha256: str = ""
    llama_cpp_server_build: str = "b10517"
    llama_cpp_timeout_seconds: float = 120
    llama_cpp_max_response_bytes: int = 65_536
    llama_cpp_allow_remote: bool = False
    llama_cpp_api_key: str = ""
    llama_cpp_api_key_file: str = ""
    deployment_profile: str = "research"
    trusted_proxy_auth_enabled: bool = False
    trusted_proxy_hmac_secret: str = ""
    trusted_proxy_replay_window_seconds: int = 60
    max_request_body_bytes: int = 22 * 1024 * 1024
    max_concurrent_requests: int = 16
    rate_limit_requests_per_minute: int = 120
    rate_limit_burst: int = 30
    metrics_enabled: bool = False
    metrics_allow_loopback: bool = True
    metrics_admin_token: str = ""
    hsts_enabled: bool = False
    anatomy_backend: str = "none"
    anatomy_required: bool = False
    anatomy_max_workers: int = 1
    contour_refinement_backend: str = "none"
    contour_refinement_device: str = "auto"
    rank03_runtime_config_path: Path | None = None
    max_agent_steps: int = 5
    max_tool_calls: int = 4
    max_expensive_vision_calls: int = 3
    agent_tool_cost_budget: int = 10

    @classmethod
    def from_env(cls) -> Settings:
        project_root = PROJECT_ROOT
        config_dir = _resolve_config_path(
            os.getenv("TBX_AGENT_CONFIG_DIR", str(project_root / "configs")), base=project_root
        )
        knowledge_dir = _resolve_config_path(
            os.getenv("TBX_AGENT_KNOWLEDGE_DIR", str(project_root / "knowledge")),
            base=project_root,
        )
        retrieval_config_path = _resolve_config_path(
            os.getenv(
                "TBX_AGENT_RETRIEVAL_CONFIG",
                str(config_dir / "retrieval.yaml"),
            ),
            base=project_root,
        )
        app_config = load_yaml(config_dir / "app.yaml")
        runtime = app_config.get("runtime", {})
        deployment = app_config.get("deployment", {})
        security = app_config.get("security", {})
        observability = app_config.get("observability", {})
        data_root = _resolve_config_path(
            os.getenv("TBX_AGENT_DATA_ROOT", str(_default_data_root())), base=project_root
        )
        db_path = _resolve_config_path(
            os.getenv("TBX_AGENT_DB_PATH", str(data_root / "tbx_agent.sqlite3")),
            base=project_root,
        )
        # Runtime case artifacts are mutable and must never share the immutable
        # model cache selected by TBX_ARTIFACT_ROOT.  Keep the older
        # TBX_AGENT_ARTIFACT_ROOT spelling only as a case-storage compatibility
        # alias; new deployments use the explicit CASE name.
        artifact_root = _resolve_config_path(
            os.getenv(
                "TBX_AGENT_CASE_ARTIFACT_ROOT",
                os.getenv("TBX_AGENT_ARTIFACT_ROOT", str(data_root / "cases")),
            ),
            base=project_root,
        )
        openai_enabled = _as_bool(
            os.getenv("TBX_AGENT_OPENAI_ENABLED"),
            bool(runtime.get("openai_enabled", False)),
        )
        configured_narrator = os.getenv("TBX_AGENT_NARRATOR_BACKEND")
        narrator_backend = (
            configured_narrator
            if configured_narrator is not None
            else runtime.get("narrator_backend") or ("openai" if openai_enabled else "none")
        )
        deployment_profile = (
            os.getenv(
                "TBX_AGENT_DEPLOYMENT_PROFILE",
                str(deployment.get("profile", "research")),
            )
            .strip()
            .lower()
        )
        if deployment_profile not in {"research", "development", "production"}:
            raise ValueError(
                "TBX_AGENT_DEPLOYMENT_PROFILE must be research, development, or production"
            )
        max_upload_bytes = int(runtime.get("max_upload_bytes", 20 * 1024 * 1024))
        return cls(
            project_root=project_root,
            config_dir=config_dir,
            knowledge_dir=knowledge_dir,
            retrieval_config_path=retrieval_config_path,
            data_root=data_root,
            db_path=db_path,
            artifact_root=artifact_root,
            fusion_policy_filename=DEFAULT_FUSION_POLICY_FILENAME,
            vision_backend=os.getenv(
                "TBX_AGENT_VISION_BACKEND", str(runtime.get("vision_backend", "mock"))
            ).strip(),
            openai_enabled=openai_enabled,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            narrator_backend=str(narrator_backend).strip().lower(),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                str(runtime.get("ollama_base_url", "http://127.0.0.1:11434")),
            ).strip(),
            ollama_model=os.getenv(
                "OLLAMA_MODEL", str(runtime.get("ollama_model", "gemma4:latest"))
            ).strip(),
            ollama_model_digest=os.getenv(
                "OLLAMA_MODEL_DIGEST", str(runtime.get("ollama_model_digest", ""))
            )
            .strip()
            .lower(),
            ollama_timeout_seconds=float(
                os.getenv(
                    "OLLAMA_TIMEOUT_SECONDS",
                    str(runtime.get("ollama_timeout_seconds", 120)),
                )
            ),
            ollama_max_response_bytes=int(
                os.getenv(
                    "OLLAMA_MAX_RESPONSE_BYTES",
                    str(runtime.get("ollama_max_response_bytes", 65536)),
                )
            ),
            ollama_allow_remote=_as_bool(
                os.getenv("TBX_AGENT_OLLAMA_ALLOW_REMOTE"),
                bool(runtime.get("ollama_allow_remote", False)),
            ),
            max_upload_bytes=max_upload_bytes,
            retain_uploaded_image=bool(runtime.get("retain_uploaded_image", True)),
            require_real_inference=_as_bool(
                os.getenv("TBX_AGENT_REQUIRE_REAL_INFERENCE"),
                bool(runtime.get("require_real_inference", False)),
            ),
            require_llm_inference=_as_bool(
                os.getenv("TBX_AGENT_REQUIRE_LLM_INFERENCE"),
                bool(runtime.get("require_llm_inference", False)),
            ),
            llama_cpp_base_url=os.getenv(
                "LLAMA_CPP_BASE_URL",
                str(runtime.get("llama_cpp_base_url", "http://127.0.0.1:11435")),
            ).strip(),
            llama_cpp_model_alias=os.getenv(
                "LLAMA_CPP_MODEL_ALIAS",
                str(
                    runtime.get(
                        "llama_cpp_model_alias",
                        "tbx-medgemma-1.5-4b-it-q4-k-m",
                    )
                ),
            ).strip(),
            llama_cpp_model_path=str(
                resolve_model_path(
                    str(runtime.get("llama_cpp_model_path", "")),
                    "LLAMA_CPP_MODEL_PATH",
                    project_root=project_root,
                )
            ),
            llama_cpp_model_sha256=os.getenv(
                "LLAMA_CPP_MODEL_SHA256", str(runtime.get("llama_cpp_model_sha256", ""))
            )
            .strip()
            .lower(),
            llama_cpp_server_build=os.getenv(
                "LLAMA_CPP_SERVER_BUILD",
                str(runtime.get("llama_cpp_server_build", "b10517")),
            ).strip(),
            llama_cpp_timeout_seconds=float(
                os.getenv(
                    "LLAMA_CPP_TIMEOUT_SECONDS",
                    str(runtime.get("llama_cpp_timeout_seconds", 120)),
                )
            ),
            llama_cpp_max_response_bytes=int(
                os.getenv(
                    "LLAMA_CPP_MAX_RESPONSE_BYTES",
                    str(runtime.get("llama_cpp_max_response_bytes", 65536)),
                )
            ),
            llama_cpp_allow_remote=_as_bool(
                os.getenv("TBX_AGENT_LLAMA_CPP_ALLOW_REMOTE"),
                bool(runtime.get("llama_cpp_allow_remote", False)),
            ),
            llama_cpp_api_key=os.getenv("LLAMA_CPP_API_KEY", "").strip(),
            llama_cpp_api_key_file=os.getenv(
                "LLAMA_CPP_API_KEY_FILE",
                str(runtime.get("llama_cpp_api_key_file", "")),
            ).strip(),
            deployment_profile=deployment_profile,
            trusted_proxy_auth_enabled=_as_bool(
                os.getenv("TBX_AGENT_TRUSTED_PROXY_AUTH_ENABLED"),
                bool(security.get("trusted_proxy_auth_enabled", False)),
            ),
            # Secrets are intentionally environment-only. They must never be
            # committed to app.yaml or returned by an API endpoint.
            trusted_proxy_hmac_secret=os.getenv("TBX_AGENT_TRUSTED_PROXY_HMAC_SECRET", ""),
            trusted_proxy_replay_window_seconds=int(
                os.getenv(
                    "TBX_AGENT_TRUSTED_PROXY_REPLAY_WINDOW_SECONDS",
                    str(security.get("trusted_proxy_replay_window_seconds", 60)),
                )
            ),
            max_request_body_bytes=int(
                os.getenv(
                    "TBX_AGENT_MAX_REQUEST_BODY_BYTES",
                    str(
                        security.get(
                            "max_request_body_bytes",
                            max_upload_bytes + 2 * 1024 * 1024,
                        )
                    ),
                )
            ),
            max_concurrent_requests=int(
                os.getenv(
                    "TBX_AGENT_MAX_CONCURRENT_REQUESTS",
                    str(security.get("max_concurrent_requests", 16)),
                )
            ),
            rate_limit_requests_per_minute=int(
                os.getenv(
                    "TBX_AGENT_RATE_LIMIT_REQUESTS_PER_MINUTE",
                    str(security.get("rate_limit_requests_per_minute", 120)),
                )
            ),
            rate_limit_burst=int(
                os.getenv(
                    "TBX_AGENT_RATE_LIMIT_BURST",
                    str(security.get("rate_limit_burst", 30)),
                )
            ),
            metrics_enabled=_as_bool(
                os.getenv("TBX_AGENT_METRICS_ENABLED"),
                bool(observability.get("metrics_enabled", False)),
            ),
            metrics_allow_loopback=_as_bool(
                os.getenv("TBX_AGENT_METRICS_ALLOW_LOOPBACK"),
                bool(observability.get("metrics_allow_loopback", True)),
            ),
            metrics_admin_token=os.getenv("TBX_AGENT_METRICS_ADMIN_TOKEN", ""),
            hsts_enabled=_as_bool(
                os.getenv("TBX_AGENT_HSTS_ENABLED"),
                bool(security.get("hsts_enabled", False)),
            ),
            anatomy_backend=os.getenv(
                "TBX_AGENT_ANATOMY_BACKEND",
                str(runtime.get("anatomy_backend", "none")),
            )
            .strip()
            .lower(),
            anatomy_required=_as_bool(
                os.getenv("TBX_AGENT_REQUIRE_ANATOMY_INFERENCE"),
                bool(runtime.get("require_anatomy_inference", False)),
            ),
            anatomy_max_workers=int(
                os.getenv(
                    "TBX_AGENT_ANATOMY_MAX_WORKERS",
                    str(runtime.get("anatomy_max_workers", 1)),
                )
            ),
            contour_refinement_backend=os.getenv(
                "TBX_AGENT_CONTOUR_REFINEMENT_BACKEND",
                str(runtime.get("contour_refinement_backend", "none")),
            )
            .strip()
            .lower(),
            contour_refinement_device=os.getenv(
                "TBX_AGENT_CONTOUR_REFINEMENT_DEVICE",
                str(runtime.get("contour_refinement_device", "auto")),
            ).strip(),
            rank03_runtime_config_path=(
                _resolve_config_path(
                    os.environ["TBX_AGENT_RANK03_RUNTIME_CONFIG"],
                    base=project_root,
                )
                if os.getenv("TBX_AGENT_RANK03_RUNTIME_CONFIG")
                else None
            ),
            max_agent_steps=int(
                os.getenv(
                    "TBX_AGENT_MAX_AGENT_STEPS",
                    str(runtime.get("max_agent_steps", 5)),
                )
            ),
            max_tool_calls=int(
                os.getenv(
                    "TBX_AGENT_MAX_TOOL_CALLS",
                    str(runtime.get("max_tool_calls", 4)),
                )
            ),
            max_expensive_vision_calls=int(
                os.getenv(
                    "TBX_AGENT_MAX_EXPENSIVE_VISION_CALLS",
                    str(runtime.get("max_expensive_vision_calls", 3)),
                )
            ),
            agent_tool_cost_budget=int(
                os.getenv(
                    "TBX_AGENT_TOOL_COST_BUDGET",
                    str(runtime.get("agent_tool_cost_budget", 10)),
                )
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.case_artifact_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def case_artifact_root(self) -> Path:
        """Mutable case-image/report root (never the immutable model cache).

        ``artifact_root`` remains the dataclass field for source compatibility
        with existing Settings constructors. New runtime code should use this
        semantic alias so it cannot be confused with ``TBX_ARTIFACT_ROOT``,
        which is reserved for immutable model artifacts.
        """

        return self.artifact_root

    def resolved_llama_cpp_api_key(self) -> str:
        """Load the local runtime credential only when llama.cpp is constructed."""

        if self.llama_cpp_api_key.strip():
            return self.llama_cpp_api_key.strip()
        return _load_llama_cpp_api_key(
            narrator_backend=self.narrator_backend,
            configured_file=self.llama_cpp_api_key_file,
            project_root=self.project_root,
        )

    def fusion_policy(self) -> dict[str, Any]:
        filename = Path(self.fusion_policy_filename)
        if (
            filename.is_absolute()
            or len(filename.parts) != 1
            or filename.name in {"", ".", ".."}
            or filename.name != self.fusion_policy_filename
        ):
            raise ValueError("fusion policy filename must be a plain filename")
        return load_json(self.config_dir / filename)

    def rank03_config(self) -> dict[str, Any]:
        path = self.rank03_runtime_config_path or self.config_dir / "rank03_runtime.json"
        return load_json(path)

    def safety_policy(self) -> dict[str, Any]:
        return load_json(self.config_dir / "safety_policy.json")


def resolve_model_path(
    configured: str,
    env_name: str,
    *,
    project_root: Path = PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environment is None else environment
    override = environment.get(env_name)
    value = str(override or configured).strip()
    prefix = "artifact://"
    if value.startswith(prefix):
        relative = PurePosixPath(value[len(prefix) :])
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(f"{env_name} contains an invalid artifact reference")
        root = default_artifact_root(environment).resolve(strict=False)
        candidate = root.joinpath(*relative.parts).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ValueError(f"{env_name} artifact reference escapes the cache root")
        return candidate
    return _resolve_config_path(value, base=project_root)
