"""Acquire immutable third-party source trees outside the Git checkout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .manager import default_artifact_root

DFINE_REPOSITORY = "https://github.com/Peterande/D-FINE.git"
DFINE_REVISION = "956d1709314c2c6a4df6f34de232054578a7449f"
DFINE_REQUIRED_FILE = Path("configs/dfine/dfine_hgnetv2_l_coco.yml")


class SourceCheckoutError(RuntimeError):
    """A pinned source checkout could not be acquired or verified."""


@dataclass(frozen=True, slots=True)
class SourceCheckoutReceipt:
    source_id: str
    repository: str
    revision: str
    destination: str
    state: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _git(*arguments: str, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise SourceCheckoutError("Git is required to acquire the pinned D-FINE source") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Git command failed").strip()
        raise SourceCheckoutError(detail) from exc
    return completed.stdout.strip()


def _verify_checkout(destination: Path, revision: str) -> None:
    if not destination.is_dir() or not (destination / ".git").is_dir():
        raise SourceCheckoutError(f"D-FINE destination is not a Git checkout: {destination}")
    observed = _git("rev-parse", "HEAD", cwd=destination).lower()
    if observed != revision.lower():
        raise SourceCheckoutError(
            f"D-FINE revision mismatch: expected {revision}, observed {observed}"
        )
    if _git("status", "--porcelain", "--untracked-files=no", cwd=destination):
        raise SourceCheckoutError("D-FINE checkout has modified tracked files")
    required = destination / DFINE_REQUIRED_FILE
    if not required.is_file():
        raise SourceCheckoutError(f"D-FINE required configuration is missing: {required}")


def verify_dfine_source(cache_dir: str | Path | None = None) -> SourceCheckoutReceipt:
    """Verify an existing checkout without network access or filesystem writes."""

    root = Path(cache_dir or default_artifact_root()).expanduser().resolve(strict=False)
    destination = (root / "sources" / "D-FINE").resolve(strict=False)
    if not destination.is_relative_to(root):
        raise SourceCheckoutError("D-FINE destination escapes the artifact root")
    _verify_checkout(destination, DFINE_REVISION)
    return SourceCheckoutReceipt(
        source_id="dfine",
        repository=DFINE_REPOSITORY,
        revision=DFINE_REVISION,
        destination=str(destination),
        state="valid_existing",
    )


def acquire_dfine_source(
    cache_dir: str | Path | None = None,
    *,
    repository: str = DFINE_REPOSITORY,
    revision: str = DFINE_REVISION,
) -> SourceCheckoutReceipt:
    """Clone and verify the pinned D-FINE revision using an atomic directory move."""

    root = Path(cache_dir or default_artifact_root()).expanduser().resolve(strict=False)
    destination = (root / "sources" / "D-FINE").resolve(strict=False)
    if not destination.is_relative_to(root):
        raise SourceCheckoutError("D-FINE destination escapes the artifact root")

    if destination.exists():
        _verify_checkout(destination, revision)
        return SourceCheckoutReceipt(
            source_id="dfine",
            repository=repository,
            revision=revision,
            destination=str(destination),
            state="valid_existing",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".D-FINE-checkout-", dir=str(destination.parent))
    ).resolve()
    try:
        _git("clone", "--filter=blob:none", "--no-checkout", repository, str(temporary))
        _git("checkout", "--detach", revision, cwd=temporary)
        _verify_checkout(temporary, revision)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
            _verify_checkout(destination, revision)
        _verify_checkout(destination, revision)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    receipt = SourceCheckoutReceipt(
        source_id="dfine",
        repository=repository,
        revision=revision,
        destination=str(destination),
        state="installed",
    )
    receipt_path = destination.parent / "D-FINE.source.json"
    receipt_path.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
