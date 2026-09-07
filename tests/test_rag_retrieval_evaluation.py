from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tbx_agent.retrieval import (
    BackendUnavailableError,
    EvaluationQuery,
    QrelCase,
    RetrievalDocument,
    RetrievedChunk,
    evaluate_retrieval_modes,
    load_evaluation_queries,
    load_qrels,
    render_benchmark_markdown,
    sha256_text,
)

PROJECT_ROOT = Path(__file__).parents[1]
EVALUATION_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_rag_retrieval.py"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_rag_index.py"
LEGACY_SMOKE_V1_ROOT = PROJECT_ROOT / "evaluation" / "retrieval" / "smoke_v1"
ACTIVE_SMOKE_ROOT = PROJECT_ROOT / "evaluation" / "retrieval" / "smoke_v5"
CORE_GUIDELINE_ROOT = PROJECT_ROOT / "evaluation" / "retrieval" / "core_guideline_seven_v2"
CDC_SUPPLEMENTAL_ROOT = PROJECT_ROOT / "evaluation" / "retrieval" / "cdc_supplemental_v2"


def _document(chunk_id: str, source_id: str) -> RetrievalDocument:
    text = f"synthetic evidence {chunk_id}"
    return RetrievalDocument(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        content_sha256=sha256_text(text),
        source_sha256=sha256_text(source_id),
        title="Synthetic",
        locator="section 1",
        jurisdiction="test",
        publication_date="2026-01-01",
        topics=("test",),
        allowed_claim_scopes=("test",),
        review_status="approved",
        retrievable=True,
    )


def test_mock_benchmark_compares_modes_metrics_latency_and_failures() -> None:
    relevant = _document("relevant", "source-relevant")
    noise = _document("noise", "source-noise")
    query = EvaluationQuery(
        schema_version=1,
        suite_id="synthetic",
        suite_version="1.0.0",
        split_id="fixed",
        query_id="q1",
        text="synthetic query",
    )
    qrel = QrelCase(
        schema_version=1,
        suite_id="synthetic",
        suite_version="1.0.0",
        corpus_generation_id="synthetic-corpus",
        query_id="q1",
        relevance={"relevant": 3},
        relevant_source_ids=("source-relevant",),
        hard_negative_chunk_ids=("noise",),
    )

    class FixedRetriever:
        def retrieve(self, _query: str, *, top_k: int):
            del top_k
            return (
                RetrievedChunk(rank=1, score=1.0, document=relevant),
                RetrievedChunk(rank=2, score=0.5, document=noise),
            )

    class FailedRetriever:
        def retrieve(self, _query: str, *, top_k: int):
            del top_k
            raise BackendUnavailableError("synthetic outage")

    ticks = iter((0, 1_000_000, 2_000_000, 5_000_000))
    report = evaluate_retrieval_modes(
        (query,),
        (qrel,),
        {"dense": FailedRetriever(), "hybrid": FixedRetriever()},
        k_values=(1, 2),
        clock_ns=lambda: next(ticks),
    )

    hybrid = report["modes"]["hybrid"]
    dense = report["modes"]["dense"]
    assert hybrid["metrics_by_k"]["1"]["recall_at_k"] == 1.0
    assert hybrid["metrics_by_k"]["1"]["mrr_at_k"] == 1.0
    assert hybrid["metrics_by_k"]["1"]["ndcg_at_k"] == 1.0
    assert hybrid["metrics_by_k"]["1"]["citation_hit_at_k"] == 1.0
    assert hybrid["latency_ms"] == {
        "p50": 3.0,
        "p95": 3.0,
        "minimum": 3.0,
        "maximum": 3.0,
    }
    assert dense["failure_counts"] == {"backend_unavailable": 1}
    assert dense["status"] == "completed_with_failures"
    markdown = render_benchmark_markdown(report)
    assert "Citation hit@K" in markdown
    assert "`backend_unavailable`: 1" in markdown


def test_checked_in_fixed_fixture_has_30_rows_and_preserves_original_seven() -> None:
    query_lines = (LEGACY_SMOKE_V1_ROOT / "queries.jsonl").read_text(encoding="utf-8")
    qrel_lines = (LEGACY_SMOKE_V1_ROOT / "qrels.jsonl").read_text(encoding="utf-8")
    queries = [json.loads(line) for line in query_lines.splitlines()]
    qrels = [json.loads(line) for line in qrel_lines.splitlines()]
    original = (
        (
            "smoke.symptoms.001",
            "肺结核可疑症状 咳嗽咳痰两周 咯血 盗汗 体重减轻",
            {"as26_symptoms": 3},
        ),
        (
            "smoke.imaging-boundary.001",
            "胸部影像异常能不能单独确诊肺结核 还需病原学检查",
            {
                "img21_confirmation_boundary": 3,
                "img21_role_boundary": 2,
                "ws288_comprehensive_diagnosis": 1,
            },
        ),
        (
            "smoke.cxr-referral.001",
            "主动筛查胸部X线异常疑似肺结核应该转诊到哪里进一步检查",
            {
                "as26_cxr_positive_referral": 3,
                "ctc21_nondesignated_referral": 1,
                "primary18_referral": 1,
            },
        ),
        (
            "smoke.tst-igra-boundary.001",
            "结核菌素皮肤试验 TST 或 IGRA 能否诊断活动性肺结核",
            {"as26_immune_positive_ltbi": 1, "who25_tst_igra_not_active": 3},
        ),
        (
            "smoke.negative-test-boundary.001",
            "痰涂片阴性或核酸检测阴性能不能排除肺结核",
            {"ats17_negative_naat_limit": 3, "ats17_smear_limitations": 3},
        ),
        (
            "smoke.reporting-referral.001",
            "非结核病定点医疗机构发现肺结核患者如何报告转诊登记",
            {"ctc21_nondesignated_referral": 3, "ctc21_reporting": 3, "ctc21_rr_referral": 1},
        ),
        (
            "smoke.treatment-product-boundary.001",
            "能否根据这次筛查结果自行停药换药或给出个人抗结核处方",
            {"policy_no_individual_treatment": 3},
        ),
    )

    assert len(queries) == len(qrels) == 30
    for query, qrel, (query_id, text, relevance) in zip(
        queries[:7], qrels[:7], original, strict=True
    ):
        assert query["query_id"] == qrel["query_id"] == query_id
        assert query["text"] == text
        assert qrel["relevance"] == relevance


def test_schema_v1_qrels_digest_and_query_shape_remain_frozen() -> None:
    queries = load_evaluation_queries(LEGACY_SMOKE_V1_ROOT / "queries.jsonl")
    qrels = load_qrels(LEGACY_SMOKE_V1_ROOT / "qrels.jsonl")

    assert all(query.guideline_scope == () and query.answerable for query in queries)
    report = evaluate_retrieval_modes(
        queries[:1],
        qrels[:1],
        {"bm25": type("Empty", (), {"retrieve": lambda self, query, *, top_k: ()})()},
        k_values=(1,),
    )
    assert report["modes"]["bm25"]["answerable_query_count"] == 1
    # Adding the v2 label field must not alter canonical v1 qrels hashing.
    from tbx_agent.retrieval.evaluation import _qrels_sha256

    assert (
        _qrels_sha256(qrels) == "e8205d27762ae4bd721e810d990910dde0269dd4be7e16ef6b6592764d71a996"
    )


def test_active_core_applies_scope_filters_and_scores_no_answer_separately() -> None:
    queries = load_evaluation_queries(CORE_GUIDELINE_ROOT / "queries.jsonl")
    qrels = load_qrels(CORE_GUIDELINE_ROOT / "qrels.jsonl")
    assert tuple(query.text for query in queries) == (
        "哪些人属于 TB 高风险人群？",
        "哪些人建议主动筛查？",
        "Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？",
        "肺结核一般怎么治疗？",
        "标准疗程大概是什么？",
        "肺结核治疗是否都必须住院，哪些情况可门诊或社区照护？",
        "怀疑肺结核时是否需要佩戴口罩？",
    )
    assert [query.answerable for query in queries] == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert [query.guideline_scope for query in queries] == [
        ("risk_groups", "active_screening"),
        ("active_screening", "risk_groups", "screening_pathway"),
        (
            "initial_diagnostic_testing",
            "drug_resistance_testing",
            "rapid_drug_resistance_testing",
            "rapid_diagnostics",
        ),
        ("treatment_education", "treatment_principles"),
        ("treatment_regimen", "standard_regimen_duration"),
        ("care_setting_education", "hospitalization_indications"),
        ("respiratory_protection",),
    ]
    relevant_ids_by_query = {
        "core.high-risk-groups.001": ("as26_high_risk", "as26_key_groups"),
        "core.active-screening-population.001": (
            "as26_key_group_path",
            "as26_key_groups",
            "as26_high_risk",
        ),
        "core.xpert-usage.001": (
            "who25_initial_lc_anaat",
            "who25_initial_rif_resistance",
        ),
        "core.treatment-principles.001": (
            "who25_ds_selection_factors",
            "who25_tb_care_monitoring",
            "who25_ds_duration_options",
        ),
        "core.standard-regimen-duration.001": ("who25_ds_duration_options",),
        "core.care-setting.001": (
            "who25_care_setting_ambulatory_majority",
            "who25_care_setting_inpatient_indications",
            "who25_care_setting_early_ambulatory_transition",
        ),
    }
    sources_by_query = {
        "core.high-risk-groups.001": "china_active_screening_2026",
        "core.active-screening-population.001": "china_active_screening_2026",
        "core.xpert-usage.001": "who_tb_diagnosis_module3_2025",
        "core.treatment-principles.001": "who_tb_treatment_module4_2025",
        "core.standard-regimen-duration.001": "who_tb_treatment_module4_2025",
        "core.care-setting.001": "who_tb_treatment_handbook_module4_2025",
    }
    relevant_by_query = {
        query_id: tuple(_document(chunk_id, sources_by_query[query_id]) for chunk_id in chunk_ids)
        for query_id, chunk_ids in relevant_ids_by_query.items()
    }
    scopes_by_query = {query.text: query.guideline_scope for query in queries}
    for query_id, documents in relevant_by_query.items():
        scope = next(query.guideline_scope[0] for query in queries if query.query_id == query_id)
        for document in documents:
            object.__setattr__(document, "allowed_claim_scopes", (scope,))

    class ScopedRetriever:
        seen: list[tuple[str, ...]] = []

        def retrieve(self, query: str, *, top_k: int, filters):
            del top_k
            self.seen.append(filters.required_claim_scopes_any)
            query_id = next(item.query_id for item in queries if item.text == query)
            return tuple(
                RetrievedChunk(rank=rank, score=1.0 / rank, document=document)
                for rank, document in enumerate(relevant_by_query.get(query_id, ()), start=1)
            )

    retriever = ScopedRetriever()
    report = evaluate_retrieval_modes(
        queries,
        qrels,
        {"bm25": retriever},
        k_values=(1, 3, 5, 10),
    )
    mode = report["modes"]["bm25"]

    assert retriever.seen == [scopes_by_query[query.text] for query in queries]
    assert mode["answerable_query_count"] == 6
    assert mode["no_answer_query_count"] == 1
    assert mode["no_answer_abstention_accuracy"] == 1.0
    assert mode["metrics_by_k"]["1"]["recall_at_k"] == pytest.approx(0.5)
    assert mode["metrics_by_k"]["10"]["mrr_at_k"] == 1.0
    assert mode["query_errors"] == []


def test_active_core_rejects_scope_leak_as_per_query_contract_failure() -> None:
    queries = load_evaluation_queries(CORE_GUIDELINE_ROOT / "queries.jsonl")[:1]
    qrels = load_qrels(CORE_GUIDELINE_ROOT / "qrels.jsonl")[:1]
    leaked = _document("leaked", "wrong-source")

    class LeakingRetriever:
        def retrieve(self, query: str, *, top_k: int, filters):
            del query, top_k, filters
            return (RetrievedChunk(rank=1, score=1.0, document=leaked),)

    report = evaluate_retrieval_modes(queries, qrels, {"dense": LeakingRetriever()})
    mode = report["modes"]["dense"]
    assert mode["failure_counts"] == {"contract_violation": 1}
    assert mode["query_errors"] == [
        {"query_id": queries[0].query_id, "errors": ["retrieval_failure"]}
    ]


def test_active_core_fixture_hashes_and_bm25_cli_report(tmp_path: Path) -> None:
    config = json.loads((CORE_GUIDELINE_ROOT / "config.json").read_text(encoding="utf-8"))
    for field, filename in (
        ("queries_sha256", "queries.jsonl"),
        ("qrels_file_sha256", "qrels.jsonl"),
    ):
        assert (
            hashlib.sha256((CORE_GUIDELINE_ROOT / filename).read_bytes()).hexdigest()
            == config[field]
        )

    output = tmp_path / "core-guideline.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATION_SCRIPT),
            "--queries",
            str(CORE_GUIDELINE_ROOT / "queries.jsonl"),
            "--qrels",
            str(CORE_GUIDELINE_ROOT / "qrels.jsonl"),
            "--modes",
            "bm25",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "completed"
    report = json.loads(output.read_text(encoding="utf-8"))
    bm25 = report["modes"]["bm25"]
    assert report["schema_version"] == 2
    assert report["answerable_query_count"] == 6
    assert report["no_answer_query_count"] == 1
    assert report["k_values"] == [1, 3, 5, 10]
    assert bm25["status"] == "completed"
    assert bm25["no_answer_abstention_accuracy"] == 1.0
    assert set(bm25["metrics_by_k"]) == {"1", "3", "5", "10"}
    assert "failed_no_answer_abstention" not in {
        error for item in bm25["query_errors"] for error in item["errors"]
    }
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    for heading in (
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        "No-answer abstention",
        "Per-query errors",
    ):
        assert heading in markdown


def test_cdc_supplemental_fixture_hashes_and_entity_level_bm25_recall(tmp_path: Path) -> None:
    config = json.loads((CDC_SUPPLEMENTAL_ROOT / "config.json").read_text(encoding="utf-8"))
    for field, filename in (
        ("queries_sha256", "queries.jsonl"),
        ("qrels_file_sha256", "qrels.jsonl"),
    ):
        assert (
            hashlib.sha256((CDC_SUPPLEMENTAL_ROOT / filename).read_bytes()).hexdigest()
            == config[field]
        )
    queries = load_evaluation_queries(CDC_SUPPLEMENTAL_ROOT / "queries.jsonl")
    qrels = load_qrels(CDC_SUPPLEMENTAL_ROOT / "qrels.jsonl")
    assert len(queries) == len(qrels) == config["query_count"] == 19
    assert all(query.answerable and query.guideline_scope for query in queries)

    output = tmp_path / "cdc-supplemental.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATION_SCRIPT),
            "--queries",
            str(CDC_SUPPLEMENTAL_ROOT / "queries.jsonl"),
            "--qrels",
            str(CDC_SUPPLEMENTAL_ROOT / "qrels.jsonl"),
            "--modes",
            "bm25",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    bm25 = report["modes"]["bm25"]
    assert report["query_count"] == 19
    assert bm25["status"] == "completed"
    assert bm25["failure_counts"] == {}
    assert bm25["metrics_by_k"]["3"]["citation_hit_at_k"] == 1.0
    assert bm25["metrics_by_k"]["5"]["recall_at_k"] == 1.0


@pytest.mark.parametrize("modes", ("bm25,dense,hybrid", "dense,hybrid"))
def test_evaluation_cli_retains_failures_and_returns_nonzero_without_loading_bge(
    tmp_path: Path, modes: str
) -> None:
    output = tmp_path / "rag-evaluation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATION_SCRIPT),
            "--modes",
            modes,
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed_with_failures"
    assert summary["modes"]["dense"] == "completed_with_failures"
    assert summary["modes"]["hybrid"] == "completed_with_failures"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["query_count"] == 31
    provenance = report["provenance"]
    assert provenance["knowledge_snapshot_id"] == "tbx-guidelines-curated-2026-09-01-v7"
    assert provenance["corpus_generation_id"] == "sparse-1b305b3eb573708ed7989a3e"
    active_config = json.loads((ACTIVE_SMOKE_ROOT / "config.json").read_text(encoding="utf-8"))
    for field, filename in (
        ("queries_sha256", "queries.jsonl"),
        ("qrels_file_sha256", "qrels.jsonl"),
    ):
        assert (
            hashlib.sha256((ACTIVE_SMOKE_ROOT / filename).read_bytes()).hexdigest()
            == (active_config[field])
        )
    assert provenance["knowledge_snapshot_id"] == active_config["expected_snapshot_id"]
    assert provenance["corpus_generation_id"] == active_config["expected_corpus_generation_id"]
    assert provenance["source_manifest_sha256"] == active_config["expected_source_manifest_sha256"]
    assert provenance["chunks_sha256"] == active_config["expected_chunks_sha256"]
    assert set(report["modes"]) == set(modes.split(","))
    if "bm25" in report["modes"]:
        assert report["modes"]["bm25"]["status"] == "completed"
    assert report["modes"]["dense"]["failure_counts"] == {"contract_violation": 31}
    assert report["modes"]["hybrid"]["failure_counts"] == {"contract_violation": 31}
    suite_validation = report["suite_validation"]
    assert suite_validation["status"] == "verified"
    assert suite_validation["discovery"] == "auto_discovered"
    assert suite_validation["seed"] == 20260901
    assert suite_validation["query_count"] == 31
    assert suite_validation["query_count_source"] == "sha256_bound_queries_file"
    assert suite_validation["retrieval_config_override_used"] is False
    assert suite_validation["retrieval_config_binding"]["content_matches"] is True
    assert report["runtime_environment"]["python"]["version"]
    assert report["runtime_environment"]["platform"]["system"]
    assert "qdrant-client" in report["runtime_environment"]["dependencies"]
    source_tree = report["source_tree_manifest"]
    assert source_tree["file_count"] > 0
    assert len(source_tree["sha256"]) == 64
    resources = report["resource_usage"]
    assert resources["process_peak_rss_bytes"] is None or (
        resources["process_peak_rss_bytes"] > 0
    )
    assert isinstance(resources["cuda_peak_vram"]["available"], bool)
    assert report["wall_time_seconds"] >= 0
    assert output.with_suffix(".md").is_file()
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "Recall@1" in markdown
    assert "Run governance and environment" in markdown
    assert "Retrieval config override used: `false`" in markdown


def test_retrieval_config_override_requires_explicit_authorization_and_is_recorded(
    tmp_path: Path,
) -> None:
    output = tmp_path / "override.json"
    base_command = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        "--queries",
        str(ACTIVE_SMOKE_ROOT / "queries.jsonl"),
        "--qrels",
        str(ACTIVE_SMOKE_ROOT / "qrels.jsonl"),
        "--suite-config",
        str(ACTIVE_SMOKE_ROOT / "config.json"),
        "--config",
        str(PROJECT_ROOT / "configs" / "retrieval_bge_m3_eval.yaml"),
        "--modes",
        "bm25",
        "--output",
        str(output),
    ]
    rejected = subprocess.run(
        base_command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert rejected.returncode == 2
    assert "retrieval config differs from the suite binding" in rejected.stderr
    assert not output.exists()

    accepted = subprocess.run(
        [*base_command, "--allow-retrieval-config-override"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert accepted.returncode == 0, accepted.stderr
    validation = json.loads(output.read_text(encoding="utf-8"))["suite_validation"]
    assert validation["retrieval_config_override_authorized"] is True
    assert validation["retrieval_config_override_used"] is True
    binding = validation["retrieval_config_binding"]
    assert binding["content_matches"] is False
    assert binding["bound_sha256"] == hashlib.sha256(
        (PROJECT_ROOT / "configs" / "retrieval.yaml").read_bytes()
    ).hexdigest()
    assert binding["actual_sha256"] == hashlib.sha256(
        (PROJECT_ROOT / "configs" / "retrieval_bge_m3_eval.yaml").read_bytes()
    ).hexdigest()
    assert "Retrieval config override used: `true`" in output.with_suffix(".md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("queries_sha256", "0" * 64, "queries SHA-256"),
        ("qrels_file_sha256", "0" * 64, "qrels SHA-256"),
        ("query_count", 30, "query_count"),
        ("seed", "20260901", "seed must be a non-negative integer"),
        ("expected_snapshot_id", "wrong-snapshot", "snapshot_id"),
        ("expected_source_manifest_sha256", "0" * 64, "source manifest SHA-256"),
        ("expected_chunks_sha256", "0" * 64, "chunks SHA-256"),
        ("expected_corpus_generation_id", "sparse-wrong", "corpus generation"),
    ),
)
def test_explicit_suite_config_rejects_identity_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    suite = json.loads((ACTIVE_SMOKE_ROOT / "config.json").read_text(encoding="utf-8"))
    suite.update(
        {
            "queries_path": str(ACTIVE_SMOKE_ROOT / "queries.jsonl"),
            "qrels_path": str(ACTIVE_SMOKE_ROOT / "qrels.jsonl"),
            "knowledge_dir": str(PROJECT_ROOT / "knowledge"),
            "retrieval_config_path": str(PROJECT_ROOT / "configs" / "retrieval.yaml"),
            "query_count": 31,
            field: value,
        }
    )
    suite_path = tmp_path / "config.json"
    suite_path.write_text(json.dumps(suite, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATION_SCRIPT),
            "--queries",
            str(ACTIVE_SMOKE_ROOT / "queries.jsonl"),
            "--qrels",
            str(ACTIVE_SMOKE_ROOT / "qrels.jsonl"),
            "--suite-config",
            str(suite_path),
            "--modes",
            "bm25",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not output.exists()


def test_rag_script_help_matches_build_and_evaluation_examples() -> None:
    build = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    evaluate = subprocess.run(
        [sys.executable, str(EVALUATION_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert build.returncode == evaluate.returncode == 0
    for option in ("--input", "--output", "--device", "--force"):
        assert option in build.stdout
    for option in (
        "--queries",
        "--qrels",
        "--suite-config",
        "--allow-retrieval-config-override",
        "--modes",
        "--output",
    ):
        assert option in evaluate.stdout
