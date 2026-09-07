from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tbx_agent.config import Settings
from tbx_agent.evaluation.trajectory import (
    ReflectionEvent,
    ResourceUsage,
    TrajectoryObservation,
    evaluate_trajectories,
    load_trajectory_suite,
    normalize_tool_receipt,
    planned_turn_to_trajectory_observation,
)
from tbx_agent.orchestration import TBXAgentGraph
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "suites" / "trajectory_v1" / "manifest.json"
CANDIDATE_ID = "synthetic-trajectory-candidate"
CANDIDATE_SHA = "c" * 64
CONTENT_SHA = "a" * 64
OUTPUT_SHA = "b" * 64


def _runtime_settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        anatomy_backend="none",
        contour_refinement_backend="none",
        narrator_backend="none",
        openai_enabled=False,
        require_real_inference=False,
        require_llm_inference=False,
    )


class _RuntimeStructuredGenerator:
    backend_id = "llama_cpp"
    model = "synthetic-qwen-planner"

    def complete_structured(self, **kwargs):
        schema_name = kwargs["schema_name"]
        if schema_name == "tbx_plan_react_plan":
            payload = {
                "goal": "回答结核病检查问题",
                "steps": [
                    {
                        "objective": "检索适用的结核病知识",
                        "evidence_need": "tb_knowledge",
                    },
                    {"objective": "整合观察并回答", "evidence_need": "none"},
                ],
            }
        elif schema_name == "tbx_agent_tool_selection":
            context = next(
                item["content"]
                for item in kwargs["messages"]
                if item.get("role") == "system"
                and "TBX_INTERNAL_CONTEXT_JSON=" in item.get("content", "")
            )
            marker = "TBX_INTERNAL_CONTEXT_JSON="
            prompt = json.loads(context[context.index(marker) + len(marker) :])
            payload = (
                {"tool": None, "direct_answer": "已根据检索观察回答。"}
                if prompt["observations"]
                else {"tool": "search_tb_knowledge", "direct_answer": None}
            )
        else:
            raise AssertionError(f"unexpected schema: {schema_name}")
        return json.dumps(payload, ensure_ascii=False), {
            "prompt_tokens": 17,
            "completion_tokens": 7,
        }

    @staticmethod
    def narrate(response):
        return response


def _tool_sequence(case) -> list[str]:
    expected = case.expected
    if expected.exact_tool_sequence is not None:
        return list(expected.exact_tool_sequence)
    return [
        slot.allowed_tools[0]
        for slot in sorted(expected.acceptable_tool_slots, key=lambda item: item.position)
    ]


def _observation(case, manifest) -> TrajectoryObservation:
    tools = _tool_sequence(case)
    is_recovery = case.expected.require_recovery
    tool_step_ids: list[str]
    middle_required = [
        step_id
        for step_id in case.expected.required_completed_steps
        if step_id not in {"route", "verify"}
    ]
    tool_step_ids = ["s1"] * len(tools) if is_recovery else middle_required[: len(tools)]
    while len(tool_step_ids) < len(tools):
        tool_step_ids.append(f"tool_{len(tool_step_ids)}")

    if is_recovery:
        plan_steps = [
            {
                "step_id": "context",
                "public_label": "构建最小必要上下文",
                "kind": "rule",
                "depends_on": [],
                "optional": False,
            },
            {
                "step_id": "s1",
                "public_label": "检索诊断与进一步检查依据",
                "kind": "tool",
                "depends_on": ["context"],
                "planned_tool": tools[0],
                "optional": False,
            },
            {
                "step_id": "verify",
                "public_label": "核验工具证据与回答",
                "kind": "verify",
                "depends_on": ["s1"],
                "optional": False,
            },
            {
                "step_id": "compose",
                "public_label": "生成受证据约束的回答",
                "kind": "synthesize",
                "depends_on": ["verify"],
                "optional": False,
            },
        ]
    else:
        plan_steps = [
            {
                "step_id": "route",
                "public_label": "确定任务边界",
                "kind": "rule",
                "depends_on": [],
                "optional": False,
            }
        ]
        previous = "route"
        for index, (tool_name, step_id) in enumerate(
            zip(tools, tool_step_ids, strict=True)
        ):
            plan_steps.append(
                {
                    "step_id": step_id,
                    "public_label": f"执行受限工具 {index + 1}",
                    "kind": "tool",
                    "depends_on": [previous],
                    "planned_tool": tool_name,
                    "optional": False,
                }
            )
            previous = step_id
        plan_steps.append(
            {
                "step_id": "verify",
                "public_label": "核验工具证据与回答",
                "kind": "verify",
                "depends_on": [previous],
                "optional": False,
            }
        )
    plan = {
        "schema_version": 1,
        "plan_id": f"{case.case_id}.plan",
        "strategy": "restricted_plan_and_solve",
        "goal_code": case.case_id,
        "planner_backend": "local_qwen",
        "max_steps": len(plan_steps),
        "allowed_tools": list(dict.fromkeys(tools)),
        "reflection_policy": "on_failure" if is_recovery else "disabled",
        "steps": plan_steps,
    }

    steps = []
    for sequence, step in enumerate(plan_steps):
        step_id = step["step_id"]
        steps.append(
            {
                "sequence_index": sequence,
                "step_id": step_id,
                "status": "recovered" if is_recovery and step_id == "s1" else "completed",
                "runtime_ms": 4.0,
                "output_sha256": OUTPUT_SHA,
            }
        )
    tool_calls = []
    for index, (tool_name, step_id) in enumerate(zip(tools, tool_step_ids, strict=True)):
        failed = is_recovery and index == 0
        tool_calls.append(
            {
                "sequence_index": 10 + index,
                "call_id": f"call-{index}",
                "step_id": step_id,
                "tool_name": tool_name,
                "status": "saturated" if failed else "succeeded",
                "requested_by": "planner" if index == 0 else "recovery",
                "runtime_ms": 20.0,
                "input_sha256": CONTENT_SHA,
                "context_sha256": CONTENT_SHA,
                "invocation_sha256": CONTENT_SHA,
                "output_sha256": OUTPUT_SHA,
                "response_sha256": OUTPUT_SHA,
                "argument_schema_valid": True,
                "output_contract_validated": True,
                "tool_contract_version": "tbx-tool-contract-v4",
                "permission": (
                    "case_read"
                    if tool_name == "get_exact_case_and_explain"
                    else "public_information"
                ),
                "requires_case": tool_name == "get_exact_case_and_explain",
                "deterministic_router": False,
                "state_mutation_allowed": False,
                "medical_decision_authority": False,
                "visual_policy_mutation_allowed": False,
                "medical_route_mutation_allowed": False,
                "fallback_used": failed,
                "error_code": "tool_capacity_exhausted" if failed else None,
            }
        )

    recoveries = []
    reflections = []
    if is_recovery:
        recoveries.append(
            {
                "sequence_index": 20,
                "trigger_call_id": "call-0",
                "strategy": "retry",
                "status": "succeeded",
                "result_step_id": "s1",
                "reason_code": "capacity_retry_succeeded",
            }
        )
        reflections.append(
            {
                "sequence_index": 21,
                "reflection_id": "reflection_1",
                "target_step_id": "s1",
                "trigger": "tool_failure",
                "decision": "retry",
                "summary_code": "capacity_retry_requested",
                "corrective_action_applied": True,
                "output_sha256": OUTPUT_SHA,
                "exposed_to_user": False,
            }
        )

    attempts = [
        {
            "sequence_index": 30 + index,
            "requested_tool": tool_name,
            "source": "user",
            "disposition": "overwritten_by_router",
        }
        for index, tool_name in enumerate(case.expected.injected_unsupported_tools)
    ]
    contexts = [
        {
            "context_id": f"ctx_{index}",
            "layer": "retrieval" if "guideline" in source_id else "turn",
            "source_id": source_id,
            "content_sha256": CONTENT_SHA,
            "provenance_status": "verified",
            "selected_for_model": True,
            "persisted_to_checkpoint": False,
            "sensitivity": "restricted",
            "trusted_as_instruction": False,
            "token_count": 20,
        }
        for index, source_id in enumerate(case.expected.required_context_source_ids)
    ]
    evidence = [
        {
            "evidence_id": evidence_id,
            "source_id": "synthetic_governed_source",
            "content_sha256": CONTENT_SHA,
            "locator": "synthetic section 1",
            "approved": True,
            "originating_call_id": "call-0" if tools else None,
        }
        for evidence_id in case.expected.required_evidence_ids
    ]
    claims = [
        {
            "claim_id": f"claim_{index}",
            "supporting_evidence_ids": [evidence_id],
            "support_status": "entailed",
        }
        for index, evidence_id in enumerate(case.expected.required_evidence_ids)
    ]
    return TrajectoryObservation.model_validate(
        {
            "schema_version": 1,
            "case_id": case.case_id,
            "suite_id": manifest.suite_id,
            "suite_version": manifest.suite_version,
            "split_hash": manifest.cases_sha256,
            "candidate_id": CANDIDATE_ID,
            "candidate_config_sha256": CANDIDATE_SHA,
            "synthetic": True,
            "clinical_validation": False,
            "plan": plan,
            "steps": steps,
            "tool_calls": tool_calls,
            "recoveries": recoveries,
            "reflections": reflections,
            "unsupported_tool_attempts": attempts,
            "contexts": contexts,
            "checkpoint": {
                "namespace_sha256": CONTENT_SHA,
                "state_sha256": OUTPUT_SHA,
                "persisted_keys": ["selected_tool", "execution_status", "receipt_sha256"],
                "contains_raw_identity": False,
                "contains_raw_message": False,
                "replay_disabled": True,
            },
            "evidence": evidence,
            "final": {
                "response_sha256": OUTPUT_SHA,
                "response_kind": case.expected.expected_response_kind or "synthetic_response",
                "public_text": "合成的受证据约束回答。",
                "citation_evidence_ids": case.expected.required_evidence_ids,
                "claims": claims,
                "asserted_tool_calls": tools,
                "stated_completed_steps": case.expected.required_completed_steps,
            },
            "usage": {
                "latency_ms": 120.0,
                "input_tokens": 200,
                "output_tokens": 80,
                "estimated_cost_usd": 0.001,
                "token_usage_measured": True,
                "cost_measured": True,
            },
        }
    )


def _report(observations=None):
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    resolved = observations or [_observation(case, manifest) for case in cases]
    return evaluate_trajectories(
        manifest=manifest,
        cases=cases,
        observations=resolved,
        candidate_id=CANDIDATE_ID,
        candidate_config_sha256=CANDIDATE_SHA,
        source_revision="synthetic-test-revision",
        suite_manifest_sha256=hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        created_at="2026-08-31T00:00:00+00:00",
    )


def test_checked_in_trajectory_suite_is_hashed_synthetic_and_separate():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)

    assert manifest.suite_version == "1.1.0"
    assert manifest.fixture_kind == "synthetic_non_clinical"
    assert manifest.selection_use is False
    assert manifest.locked_or_hidden_test_used is False
    assert len(cases) == 7
    assert {case.category for case in cases} == {
        "planning",
        "tool_selection",
        "recovery",
        "reflection",
        "injection",
        "context",
        "evidence",
    }


def test_passing_trajectories_cover_all_requested_agent_metrics():
    report = _report()

    assert report.passed, {
        result.case_id: {
            name: check.model_dump(mode="json")
            for name, check in result.checks.items()
            if not check.passed
        }
        for result in report.per_case
        if not result.passed
    }
    for metric in (
        "observation_coverage",
        "case_pass_rate",
        "plan_schema_validity_rate",
        "tool_selection_exact_rate",
        "tool_selection_acceptable_rate",
        "tool_contract_validity_rate",
        "step_completion_rate",
        "fallback_recovery_rate",
        "reflection_precision",
        "reflection_recall",
        "unsupported_tool_injection_rejection_rate",
        "context_provenance_rate",
        "secret_non_leak_rate",
        "final_evidence_faithfulness_rate",
        "execution_claim_consistency_rate",
        "resource_budget_pass_rate",
        "token_measurement_coverage",
        "cost_measurement_coverage",
        "cost_accounting_coverage",
    ):
        assert report.metrics[metric] == 1.0
        assert report.metric_fractions[metric].value == 1.0
    assert report.metrics["latency_p95_ms"] == 120.0
    assert report.metrics["total_input_tokens"] == 1400
    assert report.metrics["total_output_tokens"] == 560
    assert report.metrics["total_estimated_cost_usd"] == pytest.approx(0.007)


def test_real_graph_retry_is_adapted_from_v2_trace_without_reflection(tmp_path: Path):
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    recovery_case = next(
        case for case in cases if case.case_id == "traj.recovery.saturated.001"
    )
    service = TBXAgentService(_runtime_settings(tmp_path), max_tool_steps=3)
    service.narrator = _RuntimeStructuredGenerator()
    original_execute = service.tool_registry.execute
    first_call = True

    def saturate_once(invocation, *, fallback_factory):
        nonlocal first_call
        if first_call:
            first_call = False
            acquired = [
                service.tool_registry._capacity.acquire(blocking=False) for _ in range(4)
            ]
            assert all(acquired)
            try:
                return original_execute(invocation, fallback_factory=fallback_factory)
            finally:
                for _ in acquired:
                    service.tool_registry._capacity.release()
        return original_execute(invocation, fallback_factory=fallback_factory)

    service.tool_registry.execute = saturate_once
    graph = TBXAgentGraph(service)
    started = time.perf_counter()
    try:
        result = graph.invoke_with_receipt(
            {
                "message": "痰NAAT是什么检查？",
                "thread_id": "trajectory-runtime-thread",
                "user_id": "trajectory-runtime-user",
                "owner_scope": "tenant:trajectory-runtime",
                "case_id": None,
            }
        )
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        graph.close()

    assert result.execution_plan["source"] == "plan_react"
    assert result.execution_plan["react_steps"]
    assert [item.receipt.status.value for item in result.tool_results] == [
        "saturated",
        "succeeded",
    ]
    assert [item.receipt.selection_source for item in result.tool_results] == [
        "json_schema_fallback",
        "react_recovery",
    ]
    observation = planned_turn_to_trajectory_observation(
        result,
        trajectory_case_id=recovery_case.case_id,
        suite_id=manifest.suite_id,
        suite_version=manifest.suite_version,
        split_hash=manifest.cases_sha256,
        candidate_id=CANDIDATE_ID,
        candidate_config_sha256=CANDIDATE_SHA,
        latency_ms=elapsed_ms,
        estimated_cost_usd=0.0,
    )

    assert [call.status for call in observation.tool_calls] == ["saturated", "succeeded"]
    assert [call.requested_by for call in observation.tool_calls] == [
        "planner",
        "recovery",
    ]
    assert [call.deterministic_router for call in observation.tool_calls] == [
        item.receipt.deterministic_router for item in result.tool_results
    ]
    assert [event.strategy for event in observation.recoveries] == ["retry"]
    assert observation.recoveries[0].status == "succeeded"
    assert observation.recoveries[0].reason_code == "saturated_retry_succeeded"
    assert observation.reflections == []
    assert observation.plan is not None
    assert observation.plan["reflection_policy"] == "disabled"
    assert observation.steps[-1].step_id == "terminal"
    assert observation.steps[-1].status == "completed"

    serialized = observation.model_dump(mode="json")
    TrajectoryObservation.model_validate(serialized)
    rendered = json.dumps(serialized, ensure_ascii=False)
    for private_value in (
        "trajectory-runtime-user",
        "trajectory-runtime-thread",
        "tenant:trajectory-runtime",
        "痰NAAT是什么检查？",
        "current_query",
        "hidden_reasoning_persisted",
    ):
        assert private_value not in rendered


def test_invalid_plan_is_scored_not_rejected_and_secret_key_is_detected():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    target = observations[0]
    target.plan = dict(target.plan or {})
    target.plan["api_key"] = "not-a-real-key"

    report = _report(observations)
    result = report.per_case[0]

    assert not report.passed
    assert not result.checks["plan_schema_valid"].passed
    assert not result.checks["secret_non_leakage"].passed
    assert result.checks["secret_non_leakage"].observed["secret_key_paths"] == [
        "plan.api_key"
    ]


def test_tool_injection_execution_and_false_execution_claim_fail_closed():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    target = next(
        item
        for item in observations
        if item.case_id == "traj.injection.unsupported-tool.001"
    )
    injected = target.unsupported_tool_attempts[0]
    target.tool_calls.append(
        target.tool_calls[0].model_copy(
            update={
                "sequence_index": 31,
                "call_id": "injected-call",
                "step_id": "describe",
                "tool_name": injected.requested_tool,
            }
        )
    )

    report = _report(observations)
    result = next(item for item in report.per_case if item.case_id == target.case_id)

    assert not result.checks["unsupported_tool_injection_rejected"].passed
    assert not result.checks["tool_selection_exact"].passed
    assert not result.checks["execution_claim_consistency"].passed


def test_spurious_reflection_reduces_precision_without_changing_recall():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    target = next(
        item
        for item in observations
        if item.case_id == "traj.reflection.no-spurious.001"
    )
    target.reflections.append(
        ReflectionEvent(
            sequence_index=40,
            reflection_id="reflection_spurious",
            target_step_id="describe",
            trigger="evidence_gap",
            decision="no_change",
            summary_code="unnecessary_reflection",
            corrective_action_applied=False,
            output_sha256=OUTPUT_SHA,
            exposed_to_user=False,
        )
    )

    report = _report(observations)

    assert report.metrics["reflection_precision"] == 0.5
    assert report.metrics["reflection_recall"] == 1.0
    assert not report.passed


def test_context_canary_and_unfaithful_claim_are_detected_independently():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    target = next(item for item in observations if item.case_id == "traj.context.provenance.001")
    target.final.public_text += " SECRET_CONTEXT_CANARY_B4D2"
    target.final.claims[0].support_status = "contradicted"

    report = _report(observations)
    result = next(item for item in report.per_case if item.case_id == target.case_id)

    assert not result.checks["secret_non_leakage"].passed
    assert not result.checks["final_evidence_faithfulness"].passed
    assert report.metrics["secret_non_leak_rate"] < 1.0
    assert report.metrics["final_evidence_faithfulness_rate"] < 1.0


def test_unsafe_context_tool_contract_and_exposed_reflection_are_scored_as_failures():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    context_target = next(
        item for item in observations if item.case_id == "traj.context.provenance.001"
    )
    context_target.contexts[0].trusted_as_instruction = True
    context_target.tool_calls[0].output_contract_validated = False
    recovery_target = next(
        item for item in observations if item.case_id == "traj.recovery.saturated.001"
    )
    recovery_target.reflections[0].exposed_to_user = True

    report = _report(observations)
    context_result = next(
        item for item in report.per_case if item.case_id == context_target.case_id
    )
    recovery_result = next(
        item for item in report.per_case if item.case_id == recovery_target.case_id
    )

    assert not context_result.checks["context_provenance"].passed
    assert not context_result.checks["tool_contract_valid"].passed
    assert not recovery_result.checks["reflection_trigger_precision"].passed
    assert report.metrics["tool_contract_validity_rate"] < 1.0


def test_missing_measured_usage_fails_when_case_has_token_and_cost_budgets():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    target = observations[0]
    target.usage = target.usage.model_copy(
        update={
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
            "token_usage_measured": False,
            "cost_measured": False,
        }
    )

    report = _report(observations)
    result = report.per_case[0]

    assert not result.checks["resource_budget"].passed
    assert set(result.checks["resource_budget"].observed["failures"]) == {"tokens", "cost"}


def test_assumed_zero_local_marginal_cost_passes_budget_without_claiming_measurement():
    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observations = [_observation(case, manifest) for case in cases]
    target = observations[0]
    target.usage = ResourceUsage.model_validate(
        {
            **target.usage.model_dump(mode="json"),
            "estimated_cost_usd": 0.0,
            "cost_measured": False,
            "cost_basis": "marginal_cost_assumed_zero",
        }
    )

    report = _report(observations)
    result = report.per_case[0]

    assert result.checks["resource_budget"].passed
    assert result.checks["resource_budget"].observed["cost_basis"] == (
        "marginal_cost_assumed_zero"
    )
    assert report.metrics["cost_measurement_coverage"] == pytest.approx(6 / 7)
    assert report.metrics["cost_accounting_coverage"] == 1.0
    assert report.metrics["assumed_zero_marginal_cost_cases"] == 1
    assert report.metric_fractions["cost_measurement_coverage"].numerator == 6
    assert report.metric_fractions["cost_measurement_coverage"].denominator == 7


def test_unmeasured_remote_cost_cannot_carry_a_synthetic_zero() -> None:
    with pytest.raises(ValueError, match="unmeasured cost cannot contain"):
        ResourceUsage.model_validate(
            {
                "latency_ms": 1.0,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": 0.0,
                "token_usage_measured": False,
                "cost_measured": False,
                "cost_basis": "not_measured",
            }
        )


def test_normalize_tool_receipt_keeps_only_digest_provenance():
    normalized = normalize_tool_receipt(
        {
            "call_id": "call-1",
            "step_index": 2,
            "tool_name": "retrieve_diagnostic_guidance",
            "status": "succeeded",
            "runtime_ms": 15,
            "input_sha256": CONTENT_SHA,
            "context_sha256": CONTENT_SHA,
            "invocation_sha256": CONTENT_SHA,
            "tool_output_sha256": OUTPUT_SHA,
            "response_sha256": OUTPUT_SHA,
            "output_contract_validated": True,
            "tool_contract_version": "tbx-tool-contract-v4",
            "permission": "public_information",
            "requires_case": False,
            "deterministic_router": True,
            "state_mutation_allowed": False,
            "medical_decision_authority": False,
            "visual_policy_mutation_allowed": False,
            "medical_route_mutation_allowed": False,
            "fallback_used": False,
            "error_code": None,
            "message": "this raw field is ignored by the normalizer",
        },
        step_id="retrieve",
    )

    payload = normalized.model_dump(mode="json")
    assert payload["tool_name"] == "retrieve_diagnostic_guidance"
    assert payload["sequence_index"] == 2
    assert "message" not in payload


def test_suite_hash_drift_and_observation_binding_fail_closed(tmp_path: Path):
    manifest_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_payload["cases_sha256"] = "0" * 64
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    (tmp_path / "cases.jsonl").write_bytes(
        (MANIFEST_PATH.parent / "cases.jsonl").read_bytes()
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_trajectory_suite(bad_manifest)

    manifest, cases = load_trajectory_suite(MANIFEST_PATH)
    observation = _observation(cases[0], manifest)
    observation.split_hash = "0" * 64
    with pytest.raises(ValueError, match="suite binding mismatch"):
        evaluate_trajectories(
            manifest=manifest,
            cases=cases,
            observations=[observation],
            candidate_id=CANDIDATE_ID,
            candidate_config_sha256=CANDIDATE_SHA,
            source_revision="test",
            suite_manifest_sha256=hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        )
