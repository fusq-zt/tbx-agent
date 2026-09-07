"""Portable, traversal-safe runtime path references used by checked-in configs."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


def discover_project_root(
    environment: Mapping[str, str] | None = None,
    *,
    source_checkout: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Locate checked-in configs for editable and wheel-installed runtimes.

    A wheel's ``__file__`` lives below site-packages, so deriving the repository
    root from a fixed parent count is not portable. Containers set the explicit
    override; native source invocations can be discovered from their checkout or
    current directory.
    """

    environment = os.environ if environment is None else environment
    override = environment.get("TBX_AGENT_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    packaged_candidate = (
        source_checkout
        if source_checkout is not None
        else Path(__file__).resolve().parents[2]
    ).resolve(strict=False)
    working_candidate = (cwd if cwd is not None else Path.cwd()).resolve(strict=False)
    for candidate in dict.fromkeys((packaged_candidate, working_candidate)):
        if (candidate / "configs" / "app.yaml").is_file() and (
            candidate / "knowledge" / "source_manifest.json"
        ).is_file():
            return candidate
    return packaged_candidate


def default_runtime_root(
    environment: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
) -> Path:
    environment = os.environ if environment is None else environment
    override = (
        environment.get("TBX_AGENT_DATA_ROOT", "").strip()
        or environment.get("TBX_RUNTIME_ROOT", "").strip()
    )
    if override:
        return Path(override).expanduser()
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        local_app_data = environment.get("LOCALAPPDATA", "").strip()
        base = (
            Path(local_app_data).expanduser()
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base / "TBX-Agent" / "runtime"
    xdg_data = environment.get("XDG_DATA_HOME", "").strip()
    data_root = Path(xdg_data).expanduser() if xdg_data else Path.home() / ".local" / "share"
    return data_root / "tbx-agent"


def resolve_portable_path(
    value: str | Path,
    *,
    base: Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a local path or a checked-in ``runtime://`` reference."""

    text = str(value).strip()
    prefix = "runtime://"
    if text.startswith(prefix):
        relative = PurePosixPath(text[len(prefix) :])
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("invalid runtime path reference")
        root = default_runtime_root(environment).resolve(strict=False)
        candidate = root.joinpath(*relative.parts).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ValueError("runtime path reference escapes its root")
        return candidate
    path = Path(text).expanduser()
    return path.resolve(strict=False) if path.is_absolute() else (base / path).resolve(strict=False)
