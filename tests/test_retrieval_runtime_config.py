from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from tbx_agent.config import Settings
from tbx_agent.knowledge import RetrievalStatusCode
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_settings_defaults_to_the_project_retrieval_contract(monkeypatch) -> None:
    monkeypatch.delenv("TBX_AGENT_RETRIEVAL_CONFIG", raising=False)

    settings = Settings.from_env()

    assert settings.retrieval_config_path == (PROJECT_ROOT / "configs" / "retrieval.yaml")


def test_settings_resolves_a_relative_retrieval_contract_from_project_root(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TBX_AGENT_RETRIEVAL_CONFIG", "configs/retrieval.yaml")

    settings = Settings.from_env()

    assert settings.retrieval_config_path == (PROJECT_ROOT / "configs" / "retrieval.yaml")


def test_settings_accepts_an_external_retrieval_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_config = tmp_path / "deployment" / "retrieval.yaml"
    monkeypatch.setenv("TBX_AGENT_RETRIEVAL_CONFIG", str(external_config))

    settings = Settings.from_env()

    assert settings.retrieval_config_path == external_config.resolve()


def test_service_uses_selected_retrieval_config_without_eager_dense_load(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8")
    )
    raw["engine"]["retrieval_mode"] = "hybrid"
    raw["dense"]["enabled"] = False
    external_config = tmp_path / "retrieval.yaml"
    external_config.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    settings = replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        retrieval_config_path=external_config,
        data_root=tmp_path / "runtime",
        db_path=tmp_path / "runtime" / "state.sqlite3",
        artifact_root=tmp_path / "runtime" / "cases",
        vision_backend="mock",
        narrator_backend="none",
        openai_enabled=False,
        require_real_inference=False,
        require_llm_inference=False,
    )

    service = TBXAgentService(settings)
    try:
        assert service.retriever.retrieval_config.engine.retrieval_mode == "hybrid"
        assert service.retriever.last_runtime_state.status_code == RetrievalStatusCode.NOT_RUN
        assert service.retriever.last_runtime_state.dense_initialized is False
    finally:
        service.tool_registry.close()
        service.store.close()
