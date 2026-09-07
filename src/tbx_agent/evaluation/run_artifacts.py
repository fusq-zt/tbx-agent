"""Preserve immutable evaluation outputs before publishing convenience reports."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def require_unrecorded_destination(path: Path) -> None:
    """A named report must never mutate any previous experiment's run tree."""
    resolved = path.expanduser().resolve()
    if any((parent / "experiment.sqlite3").exists() for parent in resolved.parents):
        raise ValueError("report or ledger destination is inside an existing experiment run")


def write_snapshot(path: Path, payload: bytes) -> dict[str, str]:
    """Create an artifact once; existing run evidence is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


def publish_report(snapshot: Path, destination: Path, *, force: bool, archive_dir: Path) -> None:
    """Publish a named report while preserving any explicitly replaced version."""
    destination = destination.expanduser().resolve()
    require_unrecorded_destination(destination)
    payload = snapshot.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not force:
            raise FileExistsError(f"refusing to overwrite report: {destination}")
        previous = destination.read_bytes()
        digest = hashlib.sha256(previous).hexdigest()
        write_snapshot(archive_dir / f"{digest}-{destination.name}", previous)
    if not force:
        # Exclusive creation also protects against a competing publisher that
        # created this name after the caller's initial output preflight.
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
