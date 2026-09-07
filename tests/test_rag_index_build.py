from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tbx_agent.knowledge import GuidelineRetriever

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_rag_index.py"


def test_rag_index_dry_run_validates_without_model_or_qdrant_download() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "validated_dry_run"
    assert payload["chunk_count"] > 0
    assert payload["vector_backend"] == "qdrant_local"
    assert payload["dense_enabled"] is False
    assert len(payload["embedding_fingerprint"]) == 64


def test_rag_index_real_build_fails_closed_until_embedding_hash_is_pinned() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "dense.enabled is false" in result.stderr


@pytest.mark.parametrize("prefix_field", ["query_prefix", "document_prefix"])
def test_index_builder_and_online_config_share_prefix_identity(tmp_path: Path, prefix_field: str):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))
    fingerprints = []
    for prefix in ("", "synthetic: "):
        raw["dense"][prefix_field] = prefix
        config = tmp_path / "retrieval.yaml"
        config.write_text(yaml.safe_dump(raw), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run", "--config", str(config)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        fingerprint = json.loads(result.stdout)["embedding_fingerprint"]
        retriever = GuidelineRetriever(
            PROJECT_ROOT / "knowledge", retrieval_config_path=config
        )
        assert retriever._embedding_provenance().fingerprint == fingerprint
        fingerprints.append(fingerprint)
    assert fingerprints[0] != fingerprints[1]
