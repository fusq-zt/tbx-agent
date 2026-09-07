from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tbx_agent.artifacts.source_checkout import SourceCheckoutError, acquire_dfine_source


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "upstream"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.name", "TBX test")
    _git(source, "config", "user.email", "tbx-test@example.invalid")
    required = source / "configs" / "dfine" / "dfine_hgnetv2_l_coco.yml"
    required.parent.mkdir(parents=True)
    required.write_text("model: test\n", encoding="utf-8")
    _git(source, "add", required.relative_to(source).as_posix())
    _git(source, "commit", "-m", "fixture")
    return source, _git(source, "rev-parse", "HEAD")


def test_dfine_checkout_is_pinned_atomic_and_reusable(tmp_path: Path) -> None:
    source, revision = _source_repository(tmp_path)
    cache = tmp_path / "artifacts"

    installed = acquire_dfine_source(cache, repository=str(source), revision=revision)
    reused = acquire_dfine_source(cache, repository=str(source), revision=revision)

    assert installed.state == "installed"
    assert reused.state == "valid_existing"
    assert Path(installed.destination).is_dir()
    receipt = json.loads((cache / "sources" / "D-FINE.source.json").read_text())
    assert receipt["revision"] == revision
    assert not list((cache / "sources").glob(".D-FINE-checkout-*"))


def test_dfine_checkout_refuses_wrong_existing_revision(tmp_path: Path) -> None:
    source, revision = _source_repository(tmp_path)
    cache = tmp_path / "artifacts"
    acquire_dfine_source(cache, repository=str(source), revision=revision)

    with pytest.raises(SourceCheckoutError, match="revision mismatch"):
        acquire_dfine_source(cache, repository=str(source), revision="0" * 40)


def test_dfine_checkout_refuses_non_git_destination(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts" / "sources" / "D-FINE"
    destination.mkdir(parents=True)

    with pytest.raises(SourceCheckoutError, match="not a Git checkout"):
        acquire_dfine_source(tmp_path / "artifacts")

