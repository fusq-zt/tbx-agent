#!/usr/bin/env python3
"""Evaluate the checked-in medical dialogue matrix through the live Agent API.

This runner does not start services or provision models.  Every fixture case is
sent to ``POST /v1/agent/respond`` in a fresh thread, then checked against the
fixture's four-tool Plan+ReAct contract, retrieval-local applicability profile,
answer semantics, prohibited claims, evidence, and recovery behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = PROJECT_ROOT / "evaluation" / "fixtures" / "medical_dialogue_qa_v1.json"
REPORT_SCHEMA_VERSION = "tbx-medical-dialogue-runtime-report-v3"
API_PATH = "/v1/agent/respond"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
GUIDANCE_TOOL = "search_tb_knowledge"
HIGH_LEVEL_TOOLS = frozenset(
    {
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        GUIDANCE_TOOL,
    }
)
CASE_REQUIRED_HIGH_LEVEL_TOOLS = frozenset(
    {
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
    }
)
INTERNAL_TO_MODEL_TOOL = {
    "classify_current_cxr": "classify_cxr",
    "localize_current_cxr": "localize_cxr",
    "inspect_anatomical_context": "analyze_lung_anatomy",
    "search_tb_guidance": "search_tb_knowledge",
    "retrieve_guideline": "search_tb_knowledge",
}
EXPECTED_KEYS = {
    "task_goals",
    "scope",
    "subtopic",
    "population",
    "scenario_tags",
    "tool_names",
    "answer_status",
    "answer_contains_all",
    "answer_semantic_groups",
    "forbidden_contains",
    "required_chunk_ids",
}
LEGACY_EXPECTED_KEYS = EXPECTED_KEYS - {"scenario_tags"}


class RuntimeEvaluationError(RuntimeError):
    """Base error for invalid inputs or live-runtime failures."""


class RuntimeRequestError(RuntimeEvaluationError):
    """The Agent HTTP request did not produce a usable JSON response."""


class ResponseContractError(RuntimeEvaluationError):
    """The Agent response is missing fields required by the evaluator."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Agent API origin (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--provider",
        choices=("local_medgemma",),
        default="local_medgemma",
        help="Language-model provider sent to the Agent API.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run one fixture case; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-case HTTP timeout.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    return parser


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_base_url(value: str) -> str:
    """Return a credential-free HTTP(S) API origin/path prefix."""

    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--base-url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def agent_endpoint(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}{API_PATH}"


def load_fixture(path: Path) -> tuple[dict[str, Any], str]:
    """Load and minimally validate a medical-dialogue fixture."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read fixture: {path}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"fixture is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("fixture schema_version must be 1")
    if payload.get("clinical_validation") is not False:
        raise ValueError("fixture must explicitly declare clinical_validation=false")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture cases must be a non-empty list")
    case_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"fixture case {index} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"fixture case {index} has no case_id")
        case_ids.append(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"fixture case {case_id} has no question")
        expected = case.get("expected")
        expected_keyset = frozenset(expected) if isinstance(expected, dict) else frozenset()
        if not isinstance(expected, dict) or expected_keyset not in {
            frozenset(EXPECTED_KEYS),
            frozenset(LEGACY_EXPECTED_KEYS),
        }:
            raise ValueError(f"fixture case {case_id} has an invalid expected contract")
        is_legacy_contract = expected_keyset == frozenset(LEGACY_EXPECTED_KEYS)
        expected.setdefault("scenario_tags", [])
        for key in (
            "task_goals",
            "population",
            "scenario_tags",
            "tool_names",
            "answer_contains_all",
            "forbidden_contains",
            "required_chunk_ids",
        ):
            if not isinstance(expected[key], list) or not all(
                isinstance(item, str) for item in expected[key]
            ):
                raise ValueError(f"fixture case {case_id} expected.{key} must be strings")
        unknown_tools = sorted(set(expected["tool_names"]).difference(HIGH_LEVEL_TOOLS))
        if unknown_tools and not is_legacy_contract:
            raise ValueError(
                f"fixture case {case_id} expects tools outside the four-tool allowlist: "
                f"{unknown_tools}"
            )
        semantic_groups = expected["answer_semantic_groups"]
        if not isinstance(semantic_groups, list) or not semantic_groups:
            raise ValueError(
                f"fixture case {case_id} expected.answer_semantic_groups must be non-empty"
            )
        group_ids: list[str] = []
        for group_index, group in enumerate(semantic_groups):
            if not isinstance(group, dict) or set(group) not in (
                {"id", "any_of"},
                {"id", "all_of"},
            ):
                raise ValueError(
                    f"fixture case {case_id} semantic group {group_index} "
                    "must contain id plus exactly one of any_of or all_of"
                )
            group_id = group["id"]
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(
                    f"fixture case {case_id} semantic group {group_index} has no id"
                )
            alternative_sets = (
                [group["any_of"]] if "any_of" in group else group["all_of"]
            )
            if not isinstance(alternative_sets, list) or not alternative_sets:
                raise ValueError(
                    f"fixture case {case_id} semantic group {group_id} is empty"
                )
            for alternatives in alternative_sets:
                if (
                    not isinstance(alternatives, list)
                    or not alternatives
                    or not all(isinstance(item, str) and item.strip() for item in alternatives)
                ):
                    raise ValueError(
                        f"fixture case {case_id} semantic group {group_id} "
                        "must contain non-empty alternative-string sets"
                    )
                normalized_alternatives = [_normalized_text(item) for item in alternatives]
                if len(set(normalized_alternatives)) != len(normalized_alternatives):
                    raise ValueError(
                        f"fixture case {case_id} semantic group {group_id} "
                        "contains duplicate alternatives"
                    )
            group_ids.append(group_id)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(f"fixture case {case_id} has duplicate semantic group IDs")
    duplicates = sorted(item for item, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"fixture contains duplicate case IDs: {duplicates}")
    return payload, _sha256(raw)


def select_cases(cases: Sequence[dict[str, Any]], requested: Sequence[str]) -> list[dict[str, Any]]:
    if not requested:
        return list(cases)
    duplicates = sorted(item for item, count in Counter(requested).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate --case-id values: {duplicates}")
    by_id = {case["case_id"]: case for case in cases}
    unknown = sorted(set(requested).difference(by_id))
    if unknown:
        raise ValueError(f"unknown fixture case IDs: {unknown}")
    return [by_id[case_id] for case_id in requested]


def post_json(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """POST JSON without accepting or persisting an API key."""

    request = Request(
        endpoint,
        data=_canonical_json(payload),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "tbx-medical-dialogue-runtime-evaluator/1",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            raw_status = getattr(response, "status", None)
            status = int(raw_status if raw_status is not None else response.getcode())
            if status < 200 or status >= 300:
                raise RuntimeRequestError(f"HTTP {status} from Agent API")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeRequestError(f"HTTP {exc.code} from Agent API") from exc
    except (TimeoutError, URLError, OSError) as exc:
        reason = "request timed out" if isinstance(exc, TimeoutError) else "request failed"
        raise RuntimeRequestError(f"Agent API {reason}: {type(exc).__name__}") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeRequestError("Agent API response exceeded 4 MiB")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeRequestError("Agent API returned invalid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeRequestError("Agent API response must be a JSON object")
    return decoded


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseContractError(f"response field {field} must be an object")
    return value


def _required_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResponseContractError(f"response field {field} must be an array")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ResponseContractError(f"response field {field} must be a string or null")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    items = _required_list(value, field)
    if not all(isinstance(item, str) for item in items):
        raise ResponseContractError(f"response field {field} must contain strings")
    return list(items)


def _public_plan_steps_contract(value: Any, field: str) -> tuple[int, bool]:
    steps = _required_list(value, field)
    valid = True
    seen_ids: set[str] = set()
    for index, raw_step in enumerate(steps):
        step = _required_mapping(raw_step, f"{field}[{index}]")
        step_id = step.get("id")
        objective = step.get("objective")
        evidence_need = step.get("evidence_need")
        status = step.get("status")
        fields_valid = (
            isinstance(step_id, str)
            and bool(step_id.strip())
            and isinstance(objective, str)
            and bool(objective.strip())
            and isinstance(evidence_need, str)
            and bool(evidence_need.strip())
            and status in {"pending", "completed", "failed", "skipped"}
        )
        if not fields_valid or step_id in seen_ids:
            valid = False
        if isinstance(step_id, str):
            seen_ids.add(step_id)
    return len(steps), valid


def _model_tool_name(receipt: Mapping[str, Any], field: str) -> tuple[str, str]:
    internal_name = receipt.get("tool_name")
    if not isinstance(internal_name, str):
        raise ResponseContractError(f"{field}.tool_name must be a string")
    public_name = receipt.get("model_tool_name")
    if public_name is not None and not isinstance(public_name, str):
        raise ResponseContractError(f"{field}.model_tool_name must be a string or null")
    resolved_name = public_name or INTERNAL_TO_MODEL_TOOL.get(internal_name, internal_name)
    return str(resolved_name), internal_name


def _guidance_resolution(
    payload: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve audited retrieval dimensions without consulting ``task_spec``.

    The high-level planner only selects ``search_tb_knowledge``.  Population,
    test and scenario interpretation happens inside that tool, so evaluating
    the old planner-side dimensions would both reward the wrong architecture
    and hide retrieval-boundary errors.  The final response is an allowed
    fallback for scope/subtopic because those two fields are part of the
    public grounded-answer contract; population/scenario remain receipt-only.
    """

    guideline_receipts = [tool for tool in tools if tool["name"] == GUIDANCE_TOOL]
    receipt = guideline_receipts[-1] if guideline_receipts else None
    receipt_scope = receipt.get("resolved_scope") if receipt is not None else None
    receipt_subtopic = receipt.get("resolved_subtopic") if receipt is not None else None
    scope = receipt_scope or _optional_string(payload.get("guideline_scope"), "guideline_scope")
    subtopic = receipt_subtopic or _optional_string(
        payload.get("guideline_subtopic"), "guideline_subtopic"
    )
    return {
        "scope": scope,
        "subtopic": subtopic,
        "population": list(receipt.get("resolved_population", [])) if receipt else [],
        "scenario_tags": list(receipt.get("resolved_scenario_tags", [])) if receipt else [],
        "source": (
            "tool_receipt"
            if receipt is not None and (receipt_scope is not None or receipt_subtopic is not None)
            else "agent_response"
            if scope is not None or subtopic is not None
            else None
        ),
    }


def extract_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only public, user-visible and auditable runtime fields."""

    if not isinstance(payload, Mapping):
        raise ResponseContractError("Agent API response must be an object")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ResponseContractError("response field summary must be a non-empty string")
    trace = _required_mapping(payload.get("agent_trace"), "agent_trace")
    task_spec = _required_mapping(trace.get("task_spec"), "agent_trace.task_spec")
    task_goals = _string_list(task_spec.get("task_goals"), "task_spec.task_goals")
    raw_decisions = trace.get("decisions", [])
    if raw_decisions is None:
        raw_decisions = []
    raw_decisions = _required_list(raw_decisions, "agent_trace.decisions")
    controller_steps: list[dict[str, Any]] = []
    for index, raw_decision in enumerate(raw_decisions):
        decision = _required_mapping(raw_decision, f"agent_trace.decisions[{index}]")
        action = decision.get("action")
        step_index = decision.get("step_index")
        if not isinstance(action, str) or not isinstance(step_index, int):
            raise ResponseContractError(
                "controller decisions require string action and integer step_index"
            )
        controller_steps.append(
            {
                "step_index": step_index,
                "action": action,
                "reason_code": decision.get("reason_code"),
                "source": decision.get("source"),
            }
        )
    raw_transitions = trace.get("state_transitions", [])
    if raw_transitions is None:
        raw_transitions = []
    raw_transitions = _required_list(raw_transitions, "agent_trace.state_transitions")
    observations: list[dict[str, Any]] = []
    for index, raw_transition in enumerate(raw_transitions):
        transition = _required_mapping(
            raw_transition, f"agent_trace.state_transitions[{index}]"
        )
        step_index = transition.get("step_index")
        if not isinstance(step_index, int):
            raise ResponseContractError("state transitions require integer step_index")
        observations.append(
            {
                "step_index": step_index,
                "action": transition.get("action"),
                "tool_name": transition.get("tool_name"),
                "tool_status": transition.get("tool_status"),
                "observation_code": transition.get("observation_code"),
            }
        )
    raw_terminal = trace.get("terminal")
    terminal: dict[str, Any] | None = None
    if raw_terminal is not None:
        terminal_mapping = _required_mapping(raw_terminal, "agent_trace.terminal")
        terminal_action = terminal_mapping.get("action")
        if not isinstance(terminal_action, str):
            raise ResponseContractError("agent_trace.terminal.action must be a string")
        terminal = {
            "action": terminal_action,
            "reason_code": terminal_mapping.get("reason_code"),
        }

    receipts = _required_list(payload.get("execution_receipts"), "execution_receipts")
    tools: list[dict[str, Any]] = []
    tool_names: list[str] = []
    for index, raw_receipt in enumerate(receipts):
        receipt = _required_mapping(raw_receipt, f"execution_receipts[{index}]")
        name, internal_name = _model_tool_name(
            receipt, f"execution_receipts[{index}]"
        )
        status = receipt.get("status")
        if not isinstance(status, str):
            raise ResponseContractError("tool receipts require string status")
        resolved_population = receipt.get("resolved_population", [])
        resolved_scenarios = receipt.get("resolved_scenario_tags", [])
        if resolved_population is None:
            resolved_population = []
        if resolved_scenarios is None:
            resolved_scenarios = []
        tools.append(
            {
                "name": name,
                "internal_name": internal_name,
                "status": status,
                "attempt": receipt.get("attempt"),
                "step_index": receipt.get("step_index"),
                "plan_id": receipt.get("plan_id"),
                "step_id": receipt.get("step_id"),
                "case_id": receipt.get("case_id"),
                "requires_case": bool(receipt.get("requires_case", False)),
                "fallback_used": bool(receipt.get("fallback_used", False)),
                "error_code": receipt.get("error_code"),
                "resolved_scope": _optional_string(
                    receipt.get("resolved_guideline_scope"),
                    f"execution_receipts[{index}].resolved_guideline_scope",
                ),
                "resolved_subtopic": _optional_string(
                    receipt.get("resolved_guideline_subtopic"),
                    f"execution_receipts[{index}].resolved_guideline_subtopic",
                ),
                "resolved_population": _string_list(
                    resolved_population,
                    f"execution_receipts[{index}].resolved_population",
                ),
                "resolved_scenario_tags": _string_list(
                    resolved_scenarios,
                    f"execution_receipts[{index}].resolved_scenario_tags",
                ),
            }
        )
        if name not in tool_names:
            tool_names.append(name)

    plan = _required_mapping(payload.get("execution_plan"), "execution_plan")
    raw_execution_tool_history = _string_list(
        plan.get("tool_names"), "execution_plan.tool_names"
    )
    execution_tool_history = [
        INTERNAL_TO_MODEL_TOOL.get(name, name) for name in raw_execution_tool_history
    ]
    plan_source = plan.get("source")
    if plan_source is not None and not isinstance(plan_source, str):
        raise ResponseContractError("execution_plan.source must be a string or null")
    hidden_reasoning_persisted = plan.get("hidden_reasoning_persisted", False)
    if not isinstance(hidden_reasoning_persisted, bool):
        raise ResponseContractError(
            "execution_plan.hidden_reasoning_persisted must be boolean"
        )
    initial_plan_raw = plan.get("initial_plan")
    initial_plan: dict[str, Any] | None = None
    if initial_plan_raw is not None:
        initial_mapping = _required_mapping(
            initial_plan_raw, "execution_plan.initial_plan"
        )
        initial_step_count, initial_steps_valid = _public_plan_steps_contract(
            initial_mapping.get("steps"), "execution_plan.initial_plan.steps"
        )
        initial_plan = {
            "plan_id": initial_mapping.get("plan_id"),
            "revision": initial_mapping.get("revision"),
            "goal": initial_mapping.get("goal"),
            "step_count": initial_step_count,
            "steps_valid": initial_steps_valid,
        }
    plan_revisions_raw = plan.get("plan_revisions", [])
    if plan_revisions_raw is None:
        plan_revisions_raw = []
    plan_revisions_raw = _required_list(
        plan_revisions_raw, "execution_plan.plan_revisions"
    )
    plan_revisions: list[dict[str, Any]] = []
    for index, raw_revision in enumerate(plan_revisions_raw):
        revision = _required_mapping(
            raw_revision, f"execution_plan.plan_revisions[{index}]"
        )
        revision_step_count, revision_steps_valid = _public_plan_steps_contract(
            revision.get("steps"),
            f"execution_plan.plan_revisions[{index}].steps",
        )
        plan_revisions.append(
            {
                "revision": revision.get("revision"),
                "trigger": revision.get("trigger"),
                "reason_code": revision.get("reason_code"),
                "step_count": revision_step_count,
                "steps_valid": revision_steps_valid,
            }
        )
    react_steps_raw = plan.get("react_steps", [])
    if react_steps_raw is None:
        react_steps_raw = []
    react_steps_raw = _required_list(react_steps_raw, "execution_plan.react_steps")
    react_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(react_steps_raw):
        step = _required_mapping(raw_step, f"execution_plan.react_steps[{index}]")
        step_index = step.get("step_index")
        outcome = step.get("outcome")
        tool_name = step.get("tool_name")
        if not isinstance(step_index, int) or outcome not in {"tool_call", "answer"}:
            raise ResponseContractError(
                "ReAct steps require integer step_index and tool_call|answer outcome"
            )
        if tool_name is not None and not isinstance(tool_name, str):
            raise ResponseContractError("ReAct step tool_name must be a string or null")
        public_tool_name = (
            INTERNAL_TO_MODEL_TOOL.get(tool_name, tool_name)
            if isinstance(tool_name, str)
            else None
        )
        react_steps.append(
            {
                "step_index": step_index,
                "plan_revision": step.get("plan_revision"),
                "outcome": outcome,
                "tool_name": public_tool_name,
                "selection_mode": step.get("selection_mode"),
                "status": step.get("status"),
                "observation_code": step.get("observation_code"),
                "recovery": step.get("recovery"),
            }
        )

    evidence = _required_list(payload.get("retrieved_evidence"), "retrieved_evidence")
    chunk_ids: list[str] = []
    evidence_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(evidence):
        item = _required_mapping(raw_item, f"retrieved_evidence[{index}]")
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise ResponseContractError("retrieved evidence requires string chunk_id")
        chunk_ids.append(chunk_id)
        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        metadata = _required_mapping(metadata, f"retrieved_evidence[{index}].metadata")
        allowed_scopes = metadata.get("allowed_claim_scope", [])
        if allowed_scopes is None:
            allowed_scopes = []
        evidence_items.append(
            {
                "chunk_id": chunk_id,
                "allowed_claim_scope": _string_list(
                    allowed_scopes,
                    f"retrieved_evidence[{index}].metadata.allowed_claim_scope",
                ),
            }
        )

    narration_status = payload.get("narration_status")
    if not isinstance(narration_status, str):
        raise ResponseContractError("response field narration_status must be a string")
    narration_invoked = payload.get("narrator_generation_invoked")
    if not isinstance(narration_invoked, bool):
        raise ResponseContractError("response field narrator_generation_invoked must be boolean")
    response_case_id = _optional_string(payload.get("case_id"), "case_id")
    resolution = _guidance_resolution(payload, tools)
    return {
        "route": {
            "task_goals": list(task_goals),
            "scope": resolution["scope"],
            "subtopic": resolution["subtopic"],
            "population": resolution["population"],
            "scenario_tags": resolution["scenario_tags"],
            "guidance_resolution_source": resolution["source"],
        },
        "case_id": response_case_id,
        "controller_steps": controller_steps,
        "observations": observations,
        "terminal": terminal,
        # This is a post-run receipt summary retained for API compatibility;
        # it is deliberately not interpreted as a frozen initial plan.
        "execution_tool_history": list(execution_tool_history),
        "plan_react": {
            "source": plan_source,
            "hidden_reasoning_persisted": hidden_reasoning_persisted,
            "initial_plan": initial_plan,
            "plan_revisions": plan_revisions,
            "react_steps": react_steps,
        },
        "tool_names": tool_names,
        "tools": tools,
        "answer_status": _optional_string(payload.get("answer_status"), "answer_status"),
        "chunk_ids": chunk_ids,
        "evidence_items": evidence_items,
        "narration_status": narration_status,
        "narrator_backend": _optional_string(payload.get("narrator_backend"), "narrator_backend"),
        "narrator_model": _optional_string(payload.get("narrator_model"), "narrator_model"),
        "narrator_generation_invoked": narration_invoked,
        "answer": summary,
    }


def _normalized_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _missing_semantic_groups(
    groups: Sequence[Mapping[str, Any]], normalized_answer: str
) -> list[dict[str, Any]]:
    """Return required semantic groups for which no approved wording was found.

    A group represents one independently required proposition.  ``any_of``
    expresses whole-proposition variants.  ``all_of`` expresses compositional
    anchor slots: every slot is required, while one wording alternative within
    each slot is sufficient.  It therefore captures a relation such as
    ``negative test`` + ``does not rule out`` without requiring one exact
    sentence.  All top-level groups are required.
    """

    missing: list[dict[str, Any]] = []
    for group in groups:
        alternative_sets = (
            [list(group["any_of"])]
            if "any_of" in group
            else [list(alternatives) for alternatives in group["all_of"]]
        )
        missing_slots = [
            alternatives
            for alternatives in alternative_sets
            if not any(_normalized_text(item) in normalized_answer for item in alternatives)
        ]
        if missing_slots:
            missing.append(
                {
                    "id": group["id"],
                    "missing_alternative_sets": missing_slots,
                }
            )
    return missing


def _primary_tool_steps(tools: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Collapse retry receipts while preserving distinct controller steps."""

    steps: list[Mapping[str, Any]] = []
    seen_attempts: dict[tuple[str, str, str], set[int]] = {}
    for index, tool in enumerate(tools):
        plan_id = tool.get("plan_id")
        step_id = tool.get("step_id")
        tool_name = str(tool.get("name") or "")
        attempt = tool.get("attempt")
        if isinstance(step_id, str) and step_id:
            key = (str(plan_id or ""), step_id, tool_name)
            prior_attempts = seen_attempts.setdefault(key, set())
            if (
                isinstance(attempt, int)
                and attempt > 1
                and any(previous < attempt for previous in prior_attempts)
            ):
                prior_attempts.add(attempt)
                continue
            if isinstance(attempt, int):
                prior_attempts.add(attempt)
            steps.append(tool)
            continue
        # Without a public step ID, attempt >1 is still an explicit retry of
        # the immediately preceding same-tool invocation and is not counted as
        # a second model action.  An orphan attempt=2 remains visible as a bad
        # call instead of being silently discarded.
        if (
            isinstance(attempt, int)
            and attempt > 1
            and tools[:index]
            and str(tools[index - 1].get("name") or "") == tool_name
        ):
            continue
        steps.append(tool)
    return steps


def _trajectory_contract(
    expected: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    expected_tools = list(expected["tool_names"])
    execution_history = list(observation.get("execution_tool_history", []))
    primary_steps = _primary_tool_steps(observation.get("tools", []))
    step_names = [str(tool["name"]) for tool in primary_steps]
    expected_tool_set = set(expected_tools)
    irrelevant_steps = [name for name in step_names if name not in expected_tool_set]
    irrelevant_history = [
        name for name in execution_history if name not in expected_tool_set
    ]
    counts = Counter(step_names)
    duplicate_names = sorted(name for name, count in counts.items() if count > 1)
    duplicate_count = sum(max(0, count - 1) for count in counts.values())
    calls_by_step: Counter[int] = Counter(
        int(tool["step_index"])
        for tool in primary_steps
        if isinstance(tool.get("step_index"), int)
    )
    multi_tool_steps = sorted(
        step_index for step_index, count in calls_by_step.items() if count > 1
    )

    case_id = observation.get("case_id")
    candidate_names = list(dict.fromkeys([*execution_history, *step_names]))
    prerequisite_violations = [
        name
        for name in candidate_names
        if name in CASE_REQUIRED_HIGH_LEVEL_TOOLS and case_id is None
    ]
    prerequisite_violations.extend(
        str(tool["name"])
        for tool in primary_steps
        if (
            tool["name"] in CASE_REQUIRED_HIGH_LEVEL_TOOLS
            or tool.get("requires_case") is True
        )
        and not isinstance(tool.get("case_id"), str)
    )
    prerequisite_violations = list(dict.fromkeys(prerequisite_violations))
    plan_react = observation.get("plan_react", {})
    initial_plan = plan_react.get("initial_plan")
    plan_revisions = list(plan_react.get("plan_revisions", []))
    react_steps = list(plan_react.get("react_steps", []))
    expected_first_outcome = "tool_call" if expected_tools else "answer"
    expected_first_tool = expected_tools[0] if expected_tools else None
    first_react_step = react_steps[0] if react_steps else None
    final_architecture_case = (
        set(expected_tools) <= HIGH_LEVEL_TOOLS
        and all(
            goal in {*HIGH_LEVEL_TOOLS, "general_chat", "case_status", "social"}
            for goal in expected["task_goals"]
        )
    )
    initial_plan_valid = (
        isinstance(initial_plan, Mapping)
        and isinstance(initial_plan.get("plan_id"), str)
        and isinstance(initial_plan.get("revision"), int)
        and isinstance(initial_plan.get("goal"), str)
        and bool(str(initial_plan.get("goal", "")).strip())
        and int(initial_plan.get("step_count", 0)) >= 1
        and initial_plan.get("steps_valid") is True
    )
    first_step_matches = (
        isinstance(first_react_step, Mapping)
        and first_react_step.get("outcome") == expected_first_outcome
        and first_react_step.get("tool_name") == expected_first_tool
    )
    plan_quality_correct = (
        observation.get("route", {}).get("task_goals") == expected["task_goals"]
        and (
            plan_react.get("source") == "plan_react"
            and plan_react.get("hidden_reasoning_persisted") is False
            and initial_plan_valid
            and first_step_matches
            if final_architecture_case
            else True
        )
    )

    react_step_indexes = [int(item["step_index"]) for item in react_steps]
    react_contract_violations: list[int] = []
    if react_step_indexes != sorted(set(react_step_indexes)):
        react_contract_violations.extend(react_step_indexes)
    valid_revision_refs = {
        item
        for item in [initial_plan.get("revision") if isinstance(initial_plan, Mapping) else None]
        if isinstance(item, int)
    }
    valid_revision_refs.update(
        item["revision"]
        for item in plan_revisions
        if isinstance(item.get("revision"), int)
    )
    for item in react_steps:
        tool_name = item.get("tool_name")
        invalid = (
            item.get("outcome") == "tool_call"
            and (not isinstance(tool_name, str) or tool_name not in HIGH_LEVEL_TOOLS)
        ) or (
            item.get("outcome") == "answer" and tool_name is not None
        ) or (
            not isinstance(item.get("plan_revision"), int)
            or item.get("plan_revision") not in valid_revision_refs
        )
        if invalid:
            react_contract_violations.append(int(item["step_index"]))
    tool_react_steps = [item for item in react_steps if item.get("outcome") == "tool_call"]
    answer_react_steps = [item for item in react_steps if item.get("outcome") == "answer"]
    react_tool_names = [str(item.get("tool_name") or "") for item in tool_react_steps]
    sequence_matches_receipts = react_tool_names == step_names
    terminal_answer_valid = (
        len(answer_react_steps) == 1
        and bool(react_steps)
        and react_steps[-1].get("outcome") == "answer"
    )
    if not sequence_matches_receipts or not terminal_answer_valid:
        react_contract_violations.append(
            int(react_steps[-1]["step_index"]) if react_steps else -1
        )
    react_contract_violations.extend(multi_tool_steps)
    react_contract_violations = sorted(set(react_contract_violations))

    revision_numbers = [item.get("revision") for item in plan_revisions]
    initial_revision = initial_plan.get("revision") if isinstance(initial_plan, Mapping) else None
    revisions_valid = all(
        isinstance(item.get("revision"), int)
        and isinstance(item.get("trigger"), str)
        and bool(item["trigger"].strip())
        and isinstance(item.get("reason_code"), str)
        and bool(item["reason_code"].strip())
        and int(item.get("step_count", 0)) >= 1
        and item.get("steps_valid") is True
        for item in plan_revisions
    )
    if revision_numbers:
        revisions_valid = revisions_valid and revision_numbers == sorted(set(revision_numbers))
        if isinstance(initial_revision, int):
            revisions_valid = revisions_valid and revision_numbers[0] > initial_revision
        revisions_valid = revisions_valid and all(
            any(step.get("plan_revision") == revision for step in react_steps)
            for revision in revision_numbers
        )

    orphaned_observations: list[int] = []
    for item in tool_react_steps:
        step_index = int(item["step_index"])
        if not any(int(later["step_index"]) > step_index for later in react_steps):
            orphaned_observations.append(step_index)

    all_attempts = list(observation.get("tools", []))
    recovery_opportunities = 0
    recovered_failures = 0
    contained_failures = 0
    unhandled_failures = 0
    for index, tool in enumerate(all_attempts):
        if tool.get("status") == "succeeded":
            continue
        recovery_opportunities += 1
        later_success = any(
            later.get("name") == tool.get("name")
            and later.get("status") == "succeeded"
            for later in all_attempts[index + 1 :]
        )
        failed_step_index = tool.get("step_index")
        declared_recovery = any(
            step.get("recovery") is not None
            and step.get("recovery") is not False
            and step.get("recovery") != ""
            and step.get("recovery") != "none"
            and (
                not isinstance(failed_step_index, int)
                or int(step["step_index"]) >= failed_step_index
            )
            for step in react_steps
        )
        failed_react_steps = [
            step
            for step in tool_react_steps
            if step.get("tool_name") == tool.get("name")
            and step.get("status") != "succeeded"
        ]
        replanned_after_failure = any(
            isinstance(failed_step.get("plan_revision"), int)
            and any(
                isinstance(revision.get("revision"), int)
                and revision["revision"] > failed_step["plan_revision"]
                and any(
                    later.get("plan_revision") == revision["revision"]
                    and int(later["step_index"]) > int(failed_step["step_index"])
                    for later in react_steps
                )
                for revision in plan_revisions
            )
            for failed_step in failed_react_steps
        )
        if later_success:
            recovered_failures += 1
        elif tool.get("fallback_used") or declared_recovery or replanned_after_failure:
            contained_failures += 1
        else:
            unhandled_failures += 1
    return {
        "high_level_tool_selection_correct": (
            observation.get("route", {}).get("task_goals") == expected["task_goals"]
            and step_names == expected_tools
        ),
        "execution_history_count": len(execution_history),
        "tool_step_count": len(primary_steps),
        "irrelevant_tool_call_count": len(irrelevant_steps),
        "irrelevant_tool_names": irrelevant_steps,
        "irrelevant_history_tool_names": irrelevant_history,
        "duplicate_tool_call_count": duplicate_count,
        "duplicate_tool_names": duplicate_names,
        "multi_tool_step_violation_count": len(react_contract_violations),
        "multi_tool_step_indexes": react_contract_violations,
        "react_step_count": len(react_steps),
        "react_tool_sequence_matches_receipts": sequence_matches_receipts,
        "react_terminal_answer_valid": terminal_answer_valid,
        "react_step_cardinality_correct": not react_contract_violations,
        "prerequisite_violation_count": len(prerequisite_violations),
        "prerequisite_violation_tools": prerequisite_violations,
        "plan_quality_correct": plan_quality_correct,
        "initial_action": (
            (
                first_react_step.get("tool_name")
                if first_react_step.get("outcome") == "tool_call"
                else "answer"
            )
            if isinstance(first_react_step, Mapping)
            else None
        ),
        "expected_initial_actions": [expected_first_tool or "answer"],
        "observation_count": len(tool_react_steps),
        "plan_revision_count": len(plan_revisions),
        "plan_revisions_valid": revisions_valid,
        "orphaned_observation_count": len(orphaned_observations),
        "orphaned_observation_steps": orphaned_observations,
        "recovery_opportunity_count": recovery_opportunities,
        "recovered_failure_count": recovered_failures,
        "contained_failure_count": contained_failures,
        "unhandled_failure_count": unhandled_failures,
    }


def judge_observation(
    expected: Mapping[str, Any], observation: Mapping[str, Any]
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Judge one observation and return compact booleans plus failure details."""

    route = _required_mapping(observation.get("route"), "observation.route")
    comparisons = {
        "task_goals": (expected["task_goals"], route.get("task_goals")),
        "scope": (expected["scope"], route.get("scope")),
        "subtopic": (expected["subtopic"], route.get("subtopic")),
        "population": (expected["population"], route.get("population")),
        "scenario_tags": (expected.get("scenario_tags", []), route.get("scenario_tags")),
        "tool_names": (expected["tool_names"], observation.get("tool_names")),
        "answer_status": (expected["answer_status"], observation.get("answer_status")),
    }
    checks: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []
    for name, (wanted, actual) in comparisons.items():
        passed = actual == wanted
        checks[name] = passed
        if not passed:
            failures.append({"check": name, "expected": wanted, "actual": actual})

    normalized_answer = _normalized_text(str(observation.get("answer", "")))
    missing_semantics = _missing_semantic_groups(
        expected["answer_semantic_groups"], normalized_answer
    )
    present_forbidden = [
        fragment
        for fragment in expected["forbidden_contains"]
        if _normalized_text(fragment) in normalized_answer
    ]
    actual_chunks = set(observation.get("chunk_ids", []))
    missing_chunks = [
        chunk_id for chunk_id in expected["required_chunk_ids"] if chunk_id not in actual_chunks
    ]
    checks["answer_semantics"] = not missing_semantics
    checks["forbidden_fragments"] = not present_forbidden
    checks["required_chunk_ids"] = not missing_chunks

    trajectory = _trajectory_contract(expected, observation)
    checks["high_level_tool_selection"] = trajectory[
        "high_level_tool_selection_correct"
    ]
    checks["irrelevant_tools"] = not (
        trajectory["irrelevant_tool_call_count"]
        or trajectory["irrelevant_history_tool_names"]
    )
    checks["prerequisite_violations"] = not trajectory[
        "prerequisite_violation_count"
    ]
    checks["duplicate_tool_calls"] = not trajectory["duplicate_tool_call_count"]
    checks["react_step_cardinality"] = not trajectory[
        "multi_tool_step_violation_count"
    ]
    checks["plan_quality"] = trajectory["plan_quality_correct"]
    checks["plan_revision"] = (
        trajectory["plan_revisions_valid"]
        and not trajectory["orphaned_observation_count"]
    )
    checks["failure_recovery"] = not trajectory["unhandled_failure_count"]

    is_guidance_case = GUIDANCE_TOOL in expected["tool_names"]
    resolution_checks = [
        checks["scope"],
        checks["subtopic"],
        checks["population"],
        checks["scenario_tags"],
    ]
    checks["guideline_resolution"] = not is_guidance_case or all(resolution_checks)
    evidence_items = observation.get("evidence_items", [])
    evidence_attested = all(item.get("allowed_claim_scope") for item in evidence_items)
    status = observation.get("answer_status")
    evidence_status_consistent = (
        not observation.get("chunk_ids")
        if status == "INSUFFICIENT_EVIDENCE"
        else bool(observation.get("chunk_ids"))
        if status in {"ANSWERED", "PARTIAL"}
        else True
    )
    checks["guideline_evidence_applicability"] = (
        not is_guidance_case
        or (
            checks["guideline_resolution"]
            and checks["required_chunk_ids"]
            and checks["answer_status"]
            and evidence_attested
            and evidence_status_consistent
        )
    )
    if missing_semantics:
        failures.append({"check": "answer_semantics", "missing": missing_semantics})
    if present_forbidden:
        failures.append({"check": "forbidden_fragments", "present": present_forbidden})
    if missing_chunks:
        failures.append({"check": "required_chunk_ids", "missing": missing_chunks})
    if not checks["high_level_tool_selection"]:
        failures.append(
            {
                "check": "high_level_tool_selection",
                "expected_goals": expected["task_goals"],
                "actual_goals": route.get("task_goals"),
                "expected_tools": expected["tool_names"],
                "actual_action_sequence": [
                    item["name"]
                    for item in _primary_tool_steps(observation.get("tools", []))
                ],
            }
        )
    if not checks["irrelevant_tools"]:
        failures.append(
            {
                "check": "irrelevant_tools",
                "executed": trajectory["irrelevant_tool_names"],
                "execution_history": trajectory["irrelevant_history_tool_names"],
            }
        )
    if not checks["prerequisite_violations"]:
        failures.append(
            {
                "check": "prerequisite_violations",
                "tools": trajectory["prerequisite_violation_tools"],
            }
        )
    if not checks["duplicate_tool_calls"]:
        failures.append(
            {
                "check": "duplicate_tool_calls",
                "tools": trajectory["duplicate_tool_names"],
                "count": trajectory["duplicate_tool_call_count"],
            }
        )
    if not checks["react_step_cardinality"]:
        failures.append(
            {
                "check": "react_step_cardinality",
                "step_indexes": trajectory["multi_tool_step_indexes"],
                "tool_sequence_matches_receipts": trajectory[
                    "react_tool_sequence_matches_receipts"
                ],
                "terminal_answer_valid": trajectory["react_terminal_answer_valid"],
            }
        )
    if not checks["plan_quality"]:
        failures.append(
            {
                "check": "plan_quality",
                "initial_action": trajectory["initial_action"],
                "expected_initial_actions": trajectory["expected_initial_actions"],
            }
        )
    if not checks["plan_revision"]:
        failures.append(
            {
                "check": "plan_revision",
                "revision_records_valid": trajectory["plan_revisions_valid"],
                "orphaned_observation_steps": trajectory[
                    "orphaned_observation_steps"
                ],
            }
        )
    if not checks["failure_recovery"]:
        failures.append(
            {
                "check": "failure_recovery",
                "unhandled_failures": trajectory["unhandled_failure_count"],
            }
        )
    if not checks["guideline_resolution"]:
        failures.append(
            {
                "check": "guideline_resolution",
                "expected": {
                    "scope": expected["scope"],
                    "subtopic": expected["subtopic"],
                    "population": expected["population"],
                    "scenario_tags": expected.get("scenario_tags", []),
                },
                "actual": {
                    "scope": route.get("scope"),
                    "subtopic": route.get("subtopic"),
                    "population": route.get("population"),
                    "scenario_tags": route.get("scenario_tags"),
                    "source": route.get("guidance_resolution_source"),
                },
            }
        )
    if not checks["guideline_evidence_applicability"]:
        failures.append(
            {
                "check": "guideline_evidence_applicability",
                "evidence_attested": evidence_attested,
                "evidence_status_consistent": evidence_status_consistent,
                "actual_chunk_ids": list(observation.get("chunk_ids", [])),
            }
        )
    return checks, failures


def _error_record(
    case: Mapping[str, Any], *, latency_ms: float, error: Exception
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "category": case.get("category"),
        "question": case["question"],
        "fixture_mode": case.get("mode"),
        "status": "runtime_error",
        "latency_ms": round(latency_ms, 2),
        "observation": None,
        "quality_metrics": None,
        "checks": None,
        "failures": [
            {
                "check": "runtime",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        ],
    }


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(record["status"]) for record in records)
    check_failures = Counter(
        failure["check"]
        for record in records
        for failure in record["failures"]
        if failure["check"] != "runtime"
    )
    by_mode: dict[str, dict[str, int]] = {}
    narration_statuses: Counter[str] = Counter()
    tool_statuses: Counter[str] = Counter()
    for record in records:
        mode = str(record.get("fixture_mode") or "unspecified")
        bucket = by_mode.setdefault(
            mode, {"total": 0, "passed": 0, "failed": 0, "runtime_error": 0}
        )
        bucket["total"] += 1
        bucket[str(record["status"])] += 1
        observation = record.get("observation")
        if isinstance(observation, Mapping):
            narration_statuses[str(observation["narration_status"])] += 1
            for tool in observation["tools"]:
                tool_statuses[str(tool["status"])] += 1
    total = len(records)
    passed = statuses["passed"]
    evaluated = [
        record
        for record in records
        if isinstance(record.get("quality_metrics"), Mapping)
        and isinstance(record.get("checks"), Mapping)
    ]
    high_level_correct = sum(
        bool(record["quality_metrics"]["high_level_tool_selection_correct"])
        for record in evaluated
    )
    total_tool_steps = sum(
        int(record["quality_metrics"]["tool_step_count"]) for record in evaluated
    )
    irrelevant_calls = sum(
        int(record["quality_metrics"]["irrelevant_tool_call_count"])
        for record in evaluated
    )
    prerequisite_violations = sum(
        int(record["quality_metrics"]["prerequisite_violation_count"])
        for record in evaluated
    )
    duplicate_calls = sum(
        int(record["quality_metrics"]["duplicate_tool_call_count"])
        for record in evaluated
    )
    plan_quality_correct = sum(
        bool(record["quality_metrics"]["plan_quality_correct"])
        for record in evaluated
    )
    observation_count = sum(
        int(record["quality_metrics"]["observation_count"])
        for record in evaluated
    )
    plan_revisions = sum(
        int(record["quality_metrics"]["plan_revision_count"])
        for record in evaluated
    )
    orphaned_observations = sum(
        int(record["quality_metrics"]["orphaned_observation_count"])
        for record in evaluated
    )
    multi_tool_step_violations = sum(
        int(record["quality_metrics"]["multi_tool_step_violation_count"])
        for record in evaluated
    )
    react_cardinality_correct = sum(
        bool(record["quality_metrics"]["react_step_cardinality_correct"])
        for record in evaluated
    )
    react_steps = sum(
        int(record["quality_metrics"]["react_step_count"])
        for record in evaluated
    )
    recovery_opportunities = sum(
        int(record["quality_metrics"]["recovery_opportunity_count"])
        for record in evaluated
    )
    recovered_failures = sum(
        int(record["quality_metrics"]["recovered_failure_count"])
        for record in evaluated
    )
    contained_failures = sum(
        int(record["quality_metrics"]["contained_failure_count"])
        for record in evaluated
    )
    unhandled_failures = sum(
        int(record["quality_metrics"]["unhandled_failure_count"])
        for record in evaluated
    )
    guidance_records = [
        record
        for record in evaluated
        if GUIDANCE_TOOL in record["expected_tool_names"]
    ]
    resolution_correct = sum(
        bool(record["checks"]["guideline_resolution"])
        for record in guidance_records
    )
    evidence_applicable = sum(
        bool(record["checks"]["guideline_evidence_applicability"])
        for record in guidance_records
    )

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "total": total,
        "passed": passed,
        "failed": statuses["failed"],
        "runtime_errors": statuses["runtime_error"],
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_fixture_mode": by_mode,
        "narration_status_counts": dict(sorted(narration_statuses.items())),
        "tool_status_counts": dict(sorted(tool_statuses.items())),
        "failed_check_counts": dict(sorted(check_failures.items())),
        "agent_quality_metrics": {
            "high_level_tool_selection_accuracy": {
                "correct": high_level_correct,
                "total": len(evaluated),
                "rate": rate(high_level_correct, len(evaluated)),
            },
            "irrelevant_tool_rate": {
                "irrelevant_calls": irrelevant_calls,
                "total_tool_steps": total_tool_steps,
                "rate": rate(irrelevant_calls, total_tool_steps),
            },
            "prerequisite_violations": {
                "count": prerequisite_violations,
                "case_count": sum(
                    bool(record["quality_metrics"]["prerequisite_violation_count"])
                    for record in evaluated
                ),
            },
            "duplicate_tool_calls": {
                "count": duplicate_calls,
                "rate": rate(duplicate_calls, total_tool_steps),
            },
            "plan_quality": {
                "correct": plan_quality_correct,
                "total": len(evaluated),
                "rate": rate(plan_quality_correct, len(evaluated)),
            },
            "plan_revision": {
                "observations": observation_count,
                "revision_records": plan_revisions,
                "orphaned_observations": orphaned_observations,
                "coverage_rate": rate(
                    observation_count - orphaned_observations,
                    observation_count,
                ) if observation_count else 1.0,
            },
            "react_step_cardinality": {
                "correct_cases": react_cardinality_correct,
                "total_cases": len(evaluated),
                "compliance_rate": rate(react_cardinality_correct, len(evaluated)),
                "react_steps": react_steps,
                "tool_steps": total_tool_steps,
                "contract_violations": multi_tool_step_violations,
            },
            "failure_recovery": {
                "opportunities": recovery_opportunities,
                "recovered": recovered_failures,
                "contained_by_fallback": contained_failures,
                "unhandled": unhandled_failures,
                "recovery_or_containment_rate": rate(
                    recovered_failures + contained_failures,
                    recovery_opportunities,
                ) if recovery_opportunities else 1.0,
            },
            "guideline_resolution_accuracy": {
                "correct": resolution_correct,
                "total": len(guidance_records),
                "rate": rate(resolution_correct, len(guidance_records)),
            },
            "guideline_evidence_applicability": {
                "passed": evidence_applicable,
                "total": len(guidance_records),
                "rate": rate(evidence_applicable, len(guidance_records)),
            },
        },
    }


PostJson = Callable[[str, Mapping[str, Any], float], dict[str, Any]]


def run_evaluation(
    fixture: Mapping[str, Any],
    *,
    fixture_sha256: str,
    cases: Sequence[dict[str, Any]],
    base_url: str,
    provider: str,
    timeout_seconds: float,
    post: PostJson = post_json,
    run_id: str | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    normalized_base_url = normalize_base_url(base_url)
    endpoint = agent_endpoint(normalized_base_url)
    runtime_run_id = run_id or uuid.uuid4().hex[:16]
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        thread_id = f"medqa-{runtime_run_id}-{index:03d}"
        request_payload = {
            "thread_id": thread_id,
            "user_id": "medical-dialogue-runtime-evaluator",
            "owner_scope": f"evaluation:medical-dialogue:{runtime_run_id}",
            "message": case["question"],
            "llm_provider": provider,
        }
        case_started = time.perf_counter()
        try:
            payload = post(endpoint, request_payload, timeout_seconds)
            observation = extract_observation(payload)
            checks, failures = judge_observation(case["expected"], observation)
            quality_metrics = _trajectory_contract(case["expected"], observation)
            latency_ms = (time.perf_counter() - case_started) * 1000
            records.append(
                {
                    "case_id": case["case_id"],
                    "category": case.get("category"),
                    "question": case["question"],
                    "fixture_mode": case.get("mode"),
                    "status": "passed" if not failures else "failed",
                    "latency_ms": round(latency_ms, 2),
                    "observation": observation,
                    "expected_tool_names": list(case["expected"]["tool_names"]),
                    "quality_metrics": quality_metrics,
                    "checks": checks,
                    "failures": failures,
                }
            )
        except (RuntimeEvaluationError, ValueError, TypeError) as exc:
            latency_ms = (time.perf_counter() - case_started) * 1000
            records.append(_error_record(case, latency_ms=latency_ms, error=exc))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "suite_id": fixture.get("suite_id"),
        "fixture_sha256": fixture_sha256,
        "clinical_validation": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "provider": provider,
            "base_url": normalized_base_url,
            "endpoint_path": API_PATH,
            "timeout_seconds": timeout_seconds,
            "run_id": runtime_run_id,
            "elapsed_ms": round(elapsed_ms, 2),
        },
        "summary": _summarize(records),
        "cases": records,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"output already exists; choose a new report path: {path}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        fixture, fixture_sha256 = load_fixture(args.fixture)
        cases = select_cases(fixture["cases"], args.case_id)
        report = run_evaluation(
            fixture,
            fixture_sha256=fixture_sha256,
            cases=cases,
            base_url=args.base_url,
            provider=args.provider,
            timeout_seconds=args.timeout_seconds,
        )
        _write_report(args.output, report)
    except (OSError, ValueError, RuntimeEvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(
        "medical dialogue runtime: "
        f"{summary['passed']}/{summary['total']} passed; "
        f"{summary['failed']} failed; {summary['runtime_errors']} runtime errors"
    )
    return 0 if summary["failed"] == 0 and summary["runtime_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
