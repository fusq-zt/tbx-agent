from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import BoundedSemaphore, Lock, RLock

from ..schemas import AgentResponse, GuidelineAnswerStatus, ResponseKind
from .contracts import (
    TOOL_CONTRACT_VERSION,
    ToolAvailability,
    ToolCallStatus,
    ToolCostTier,
    ToolErrorCategory,
    ToolInvocation,
    ToolName,
    ToolOutcome,
    ToolPermission,
    ToolReceipt,
    ToolResult,
    ToolStatus,
    ToolUnavailableError,
)

ToolHandler = Callable[[ToolInvocation], AgentResponse]
ToolHealthCheck = Callable[[], bool]
ToolFallbackFactory = Callable[[ToolInvocation, ToolCallStatus, str], AgentResponse]

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Static least-privilege contract for one allowlisted handler."""

    name: str
    audit_action: str
    handler: ToolHandler
    timeout_seconds: float
    permission: ToolPermission = ToolPermission.PUBLIC_INFORMATION
    allowed_response_kinds: frozenset[ResponseKind] = frozenset({ResponseKind.SAFE_ABSTENTION})
    requires_case: bool = False
    state_mutation_allowed: bool = False
    cost_tier: ToolCostTier = ToolCostTier.LOW
    cost_units: int = 1
    expensive_vision: bool = False
    health_check: ToolHealthCheck | None = None

    def __post_init__(self) -> None:
        if not _TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("tool name must be a lowercase allowlist identifier")
        if not self.audit_action:
            raise ValueError("tool audit action is required")
        if self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive")
        if not self.allowed_response_kinds:
            raise ValueError("tool must declare at least one allowed response kind")
        if ResponseKind.SAFE_ABSTENTION not in self.allowed_response_kinds:
            raise ValueError("tool response contract must allow safe abstention")
        if self.cost_units < 0 or self.cost_units > 10:
            raise ValueError("tool cost units must be in [0, 10]")


@dataclass(slots=True)
class _RuntimeState:
    last_call_status: ToolCallStatus | None = None
    consecutive_failures: int = 0
    last_runtime_ms: int | None = None
    last_error_code: str | None = None


class ToolRegistry:
    """Allowlisted executor with bounded capacity, deadlines, and safe degradation.

    Handlers are deterministic application functions with an explicit least-privilege
    contract. The narrator never supplies names or parameters. A timed-out Python
    thread cannot be force-killed,
    so the semaphore remains held until it actually exits; this prevents timed-out
    work from accumulating in an unbounded executor queue.
    """

    def __init__(self, *, max_steps: int = 1, max_workers: int = 4):
        if max_steps < 1 or max_steps > 8:
            raise ValueError("max_steps must be in [1, 8]")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_steps = max_steps
        self._definitions: dict[str, ToolDefinition] = {}
        self._runtime_states: dict[str, _RuntimeState] = {}
        self._state_lock = RLock()
        self._capacity = BoundedSemaphore(max_workers)
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tbx-tool")

    def register(self, definition: ToolDefinition) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("cannot register a tool after registry shutdown")
            if definition.name in self._definitions:
                raise ValueError(f"duplicate tool registration: {definition.name}")
            self._definitions[definition.name] = definition
            self._runtime_states[definition.name] = _RuntimeState()

    def names(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(sorted(self._definitions))

    def statuses(self) -> list[ToolStatus]:
        with self._state_lock:
            definitions = tuple(self._definitions.values())
            registry_closed = self._closed
        statuses: list[ToolStatus] = []
        for definition in sorted(definitions, key=lambda item: item.name):
            availability = (
                ToolAvailability.UNAVAILABLE if registry_closed else ToolAvailability.READY
            )
            detail_code = "tool_registry_closed" if registry_closed else None
            if availability == ToolAvailability.READY and definition.health_check is not None:
                try:
                    ready = bool(definition.health_check())
                except Exception:
                    ready = False
                    detail_code = "health_check_failed"
                if not ready:
                    availability = ToolAvailability.UNAVAILABLE
                    detail_code = detail_code or "health_check_not_ready"
            with self._state_lock:
                runtime = self._runtime_states[definition.name]
                last_call_status = runtime.last_call_status
                consecutive_failures = runtime.consecutive_failures
                last_runtime_ms = runtime.last_runtime_ms
                runtime_error_code = runtime.last_error_code
            if availability == ToolAvailability.READY and consecutive_failures > 0:
                availability = ToolAvailability.DEGRADED
                detail_code = runtime_error_code
            statuses.append(
                ToolStatus(
                    name=definition.name,
                    availability=availability,
                    timeout_ms=max(1, round(definition.timeout_seconds * 1000)),
                    max_steps=self.max_steps,
                    permission=definition.permission,
                    requires_case=definition.requires_case,
                    state_mutation_allowed=definition.state_mutation_allowed,
                    cost_tier=definition.cost_tier,
                    cost_units=definition.cost_units,
                    expensive_vision=definition.expensive_vision,
                    allowed_response_kinds=sorted(
                        definition.allowed_response_kinds, key=lambda item: item.value
                    ),
                    detail_code=detail_code,
                    last_call_status=last_call_status,
                    consecutive_failures=consecutive_failures,
                    last_runtime_ms=last_runtime_ms,
                )
            )
        return statuses

    def execute(
        self,
        invocation: ToolInvocation,
        *,
        fallback_factory: ToolFallbackFactory,
    ) -> ToolResult:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        call_id = str(uuid.uuid4())
        hashes = self._invocation_hashes(invocation)
        with self._state_lock:
            definition = self._definitions.get(invocation.tool_name)
            registry_closed = self._closed

        fallback_args = {
            "invocation": invocation,
            "definition": definition,
            "fallback_factory": fallback_factory,
            "call_id": call_id,
            **hashes,
            "started_at": started_at,
            "started": started,
        }

        if invocation.max_steps != self.max_steps:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.REJECTED,
                error_code="tool_step_policy_mismatch",
                error_type=None,
            )
        if invocation.step_index >= self.max_steps:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.STEP_LIMIT_EXCEEDED,
                error_code="tool_step_limit_exceeded",
                error_type=None,
            )
        if definition is None:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.UNAVAILABLE,
                error_code="tool_not_allowlisted",
                error_type=None,
            )
        if definition.requires_case and invocation.case_id is None:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.REJECTED,
                error_code="tool_case_context_required",
                error_type=None,
            )
        if registry_closed:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.UNAVAILABLE,
                error_code="tool_registry_closed",
                error_type=None,
            )
        if definition.health_check is not None:
            try:
                available = bool(definition.health_check())
            except Exception:
                available = False
            if not available:
                return self._fallback_result(
                    **fallback_args,
                    status=ToolCallStatus.UNAVAILABLE,
                    error_code="tool_health_check_not_ready",
                    error_type=None,
                )
        if not self._capacity.acquire(blocking=False):
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.SATURATED,
                error_code="tool_capacity_exhausted",
                error_type=None,
            )

        try:
            future = self._executor.submit(definition.handler, invocation)
        except Exception:
            self._capacity.release()
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.UNAVAILABLE,
                error_code="tool_executor_unavailable",
                error_type=None,
            )
        release_lock = Lock()
        capacity_released = False

        def release_capacity_once() -> None:
            nonlocal capacity_released
            with release_lock:
                if capacity_released:
                    return
                capacity_released = True
                self._capacity.release()

        future.add_done_callback(lambda _future: release_capacity_once())

        try:
            response = future.result(timeout=definition.timeout_seconds)
            # ``Future.result`` may wake before done callbacks finish. Release
            # synchronously as well so a burst of fast sequential calls cannot
            # falsely exhaust every worker slot; the guarded callback still
            # releases timed-out work only when its thread actually exits.
            release_capacity_once()
        except FutureTimeoutError:
            future.cancel()
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.TIMED_OUT,
                error_code="tool_timeout",
                error_type="TimeoutError",
            )
        except (KeyError, PermissionError):
            # Caller/domain errors preserve established API 404/403 semantics.
            raise
        except ToolUnavailableError as exc:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.UNAVAILABLE,
                error_code="tool_dependency_unavailable",
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.FAILED,
                error_code="tool_execution_failed",
                error_type=type(exc).__name__,
            )

        contract_error = self._response_contract_error(
            response, invocation=invocation, definition=definition
        )
        if contract_error is not None:
            return self._fallback_result(
                **fallback_args,
                status=ToolCallStatus.FAILED,
                error_code=contract_error,
                error_type=type(response).__name__,
            )

        receipt = self._receipt(
            invocation=invocation,
            response=response,
            definition=definition,
            status=ToolCallStatus.SUCCEEDED,
            fallback_used=False,
            error_code=None,
            error_type=None,
            call_id=call_id,
            **hashes,
            started_at=started_at,
            started=started,
        )
        result = ToolResult(
            response=response,
            receipt=receipt,
            audit_action=definition.audit_action,
        )
        self._record_runtime(result)
        return result

    def close(self, *, wait: bool = False) -> None:
        with self._state_lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    @staticmethod
    def _invocation_hashes(invocation: ToolInvocation) -> dict[str, str]:
        input_sha256 = hashlib.sha256(invocation.message.encode("utf-8")).hexdigest()
        context_sha256 = _canonical_sha256(
            {
                "case_id": invocation.case_id,
                "owner_scope": invocation.owner_scope,
                "thread_id": invocation.thread_id,
                "user_id": invocation.user_id,
            }
        )
        invocation_sha256 = _canonical_sha256(
            {
                "case_id": invocation.case_id,
                "context_sha256": context_sha256,
                "input_sha256": input_sha256,
                "max_steps": invocation.max_steps,
                "plan_id": invocation.plan_id,
                "request_id": invocation.request_id,
                "routing_policy_id": invocation.routing_policy_id,
                "safety_policy_id": invocation.safety_policy_id,
                "step_index": invocation.step_index,
                "step_id": invocation.step_id,
                "selection_source": invocation.selection_source,
                "attempt": invocation.attempt,
                "max_attempts": invocation.max_attempts,
                "idempotency_key": invocation.idempotency_key,
                "tool_contract_version": TOOL_CONTRACT_VERSION,
                "tool_name": invocation.tool_name,
                "model_tool_name": invocation.model_tool_name,
                "trace_id": invocation.trace_id,
            }
        )
        return {
            "input_sha256": input_sha256,
            "context_sha256": context_sha256,
            "invocation_sha256": invocation_sha256,
        }

    def _fallback_result(
        self,
        *,
        invocation: ToolInvocation,
        definition: ToolDefinition | None,
        fallback_factory: ToolFallbackFactory,
        status: ToolCallStatus,
        error_code: str,
        error_type: str | None,
        call_id: str,
        input_sha256: str,
        context_sha256: str,
        invocation_sha256: str,
        started_at: datetime,
        started: float,
    ) -> ToolResult:
        output_contract_validated = True
        try:
            response = fallback_factory(invocation, status, error_code)
            fallback_error = self._fallback_contract_error(response, invocation=invocation)
            if fallback_error is not None:
                raise ValueError(fallback_error)
        except Exception:
            output_contract_validated = False
            response = self._last_resort_fallback(invocation)
        timeout_seconds = definition.timeout_seconds if definition is not None else 1.0
        receipt = self._receipt(
            invocation=invocation,
            response=response,
            definition=definition,
            status=status,
            fallback_used=True,
            error_code=error_code,
            error_type=error_type,
            call_id=call_id,
            input_sha256=input_sha256,
            context_sha256=context_sha256,
            invocation_sha256=invocation_sha256,
            started_at=started_at,
            started=started,
            fallback_timeout_seconds=timeout_seconds,
            output_contract_validated=output_contract_validated,
        )
        result = ToolResult(
            response=response,
            receipt=receipt,
            audit_action="tool_execution_degraded",
        )
        self._record_runtime(result)
        return result

    def _record_runtime(self, result: ToolResult) -> None:
        tool_name = result.receipt.tool_name
        if tool_name not in self._runtime_states:
            return
        with self._state_lock:
            runtime = self._runtime_states[tool_name]
            runtime.last_call_status = result.receipt.status
            runtime.last_runtime_ms = result.receipt.runtime_ms
            if result.receipt.status == ToolCallStatus.SUCCEEDED:
                runtime.consecutive_failures = 0
                runtime.last_error_code = None
            elif result.receipt.status in {
                ToolCallStatus.TIMED_OUT,
                ToolCallStatus.FAILED,
                ToolCallStatus.UNAVAILABLE,
                ToolCallStatus.SATURATED,
            }:
                runtime.consecutive_failures += 1
                runtime.last_error_code = result.receipt.error_code

    def _receipt(
        self,
        *,
        invocation: ToolInvocation,
        response: AgentResponse,
        definition: ToolDefinition | None,
        status: ToolCallStatus,
        fallback_used: bool,
        error_code: str | None,
        error_type: str | None,
        call_id: str,
        input_sha256: str,
        context_sha256: str,
        invocation_sha256: str,
        started_at: datetime,
        started: float,
        fallback_timeout_seconds: float | None = None,
        output_contract_validated: bool = True,
    ) -> ToolReceipt:
        finished_at = datetime.now(UTC)
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        timeout_seconds = (
            definition.timeout_seconds
            if definition is not None
            else fallback_timeout_seconds or 1.0
        )
        output_sha256 = _canonical_sha256(response.model_dump(mode="json"))
        error_category = self._error_category(status=status, error_code=error_code)
        outcome = self._outcome(
            invocation=invocation,
            response=response,
            fallback_used=fallback_used,
        )
        observation_code = self._observation_code(
            invocation=invocation,
            response=response,
            fallback_used=fallback_used,
        )
        return ToolReceipt(
            call_id=call_id,
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            tool_name=invocation.tool_name,
            model_tool_name=invocation.model_tool_name,
            routing_policy_id=invocation.routing_policy_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            runtime_ms=elapsed_ms,
            timeout_ms=max(1, round(timeout_seconds * 1000)),
            step_index=invocation.step_index,
            max_steps=invocation.max_steps,
            plan_id=invocation.plan_id,
            step_id=invocation.step_id,
            selection_source=invocation.selection_source,
            attempt=invocation.attempt,
            max_attempts=invocation.max_attempts,
            idempotency_key=invocation.idempotency_key,
            input_sha256=input_sha256,
            context_sha256=context_sha256,
            invocation_sha256=invocation_sha256,
            tool_output_sha256=output_sha256,
            response_sha256=output_sha256,
            case_id=invocation.case_id,
            response_kind=response.response_kind,
            citation_count=len(response.citations),
            fallback_used=fallback_used,
            error_code=error_code,
            error_type=error_type,
            error_category=error_category,
            retryable=status == ToolCallStatus.SATURATED,
            outcome=outcome,
            permission=(
                definition.permission
                if definition is not None
                else ToolPermission.PUBLIC_INFORMATION
            ),
            requires_case=definition.requires_case if definition is not None else False,
            output_contract_validated=output_contract_validated,
            deterministic_router=invocation.selection_source != "llm_plan_and_solve",
            state_mutation_allowed=(
                definition.state_mutation_allowed if definition is not None else False
            ),
            cost_tier=(definition.cost_tier if definition is not None else ToolCostTier.LOW),
            cost_units=definition.cost_units if definition is not None else 0,
            expensive_vision=(definition.expensive_vision if definition is not None else False),
            observation_code=observation_code,
            resolved_guideline_scope=invocation.guideline_scope,
            resolved_guideline_subtopic=invocation.subtopic,
            resolved_population=list(invocation.population),
            resolved_product_terms=list(invocation.product_terms),
            resolved_scenario_tags=list(invocation.scenario_tags),
        )

    @staticmethod
    def _error_category(
        *,
        status: ToolCallStatus,
        error_code: str | None,
    ) -> ToolErrorCategory | None:
        if status == ToolCallStatus.SUCCEEDED:
            return None
        if status == ToolCallStatus.TIMED_OUT:
            return ToolErrorCategory.TIMEOUT
        if status == ToolCallStatus.SATURATED:
            return ToolErrorCategory.CAPACITY
        if error_code and error_code.startswith("tool_output_"):
            return ToolErrorCategory.CONTRACT
        if status in {ToolCallStatus.REJECTED, ToolCallStatus.STEP_LIMIT_EXCEEDED}:
            return ToolErrorCategory.POLICY
        if status == ToolCallStatus.UNAVAILABLE:
            return ToolErrorCategory.DEPENDENCY
        return ToolErrorCategory.DEPENDENCY

    @staticmethod
    def _outcome(
        *,
        invocation: ToolInvocation,
        response: AgentResponse,
        fallback_used: bool,
    ) -> ToolOutcome:
        if fallback_used:
            return ToolOutcome.EXECUTION_DEGRADED
        # Coverage status is the semantic result of a guideline lookup.  It must
        # win over both the tool identity and the presentation-oriented response
        # kind: treatment and other education responses can still be explicit
        # evidence gaps.
        if response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE:
            return ToolOutcome.EVIDENCE_GAP
        if invocation.tool_name == ToolName.EMERGENCY_TRIAGE.value:
            return ToolOutcome.EMERGENCY_HANDOFF
        if invocation.tool_name == ToolName.DESCRIBE_AGENT_CAPABILITIES.value:
            return ToolOutcome.CAPABILITY_ONLY
        if invocation.tool_name in {
            ToolName.GET_EXACT_CASE_AND_EXPLAIN.value,
            ToolName.CLASSIFY_CURRENT_CXR.value,
            ToolName.LOCALIZE_CURRENT_CXR.value,
            ToolName.INSPECT_ANATOMICAL_CONTEXT.value,
            ToolName.INSPECT_IMAGE_QUALITY.value,
        }:
            return ToolOutcome.CASE_EVIDENCE_READY
        if response.response_kind == ResponseKind.SAFE_ABSTENTION:
            return ToolOutcome.EVIDENCE_GAP
        return ToolOutcome.ANSWER_READY

    @staticmethod
    def _observation_code(
        *,
        invocation: ToolInvocation,
        response: AgentResponse,
        fallback_used: bool,
    ) -> str:
        if fallback_used:
            return "tool_execution_failed"
        codes = {
            ToolName.EMERGENCY_TRIAGE.value: "emergency_handoff_completed",
            ToolName.GET_EXACT_CASE_AND_EXPLAIN.value: "classification_evidence_read",
            ToolName.CLASSIFY_CURRENT_CXR.value: "classification_state_updated",
            ToolName.LOCALIZE_CURRENT_CXR.value: "localization_state_updated",
            ToolName.INSPECT_ANATOMICAL_CONTEXT.value: "anatomy_state_updated",
            ToolName.INSPECT_IMAGE_QUALITY.value: "quality_evidence_read",
            ToolName.COMPARE_WITH_PRIOR_CXR.value: "prior_image_not_found",
            ToolName.DESCRIBE_AGENT_CAPABILITIES.value: "capabilities_described",
        }
        if invocation.tool_name in codes:
            return codes[invocation.tool_name]
        if invocation.tool_name == ToolName.SEARCH_TB_GUIDANCE.value:
            return {
                "ANSWERED": "guideline_answered",
                "PARTIAL": "guideline_partial",
                "INSUFFICIENT_EVIDENCE": "guideline_insufficient_evidence",
            }.get(
                response.answer_status.value if response.answer_status is not None else "",
                "guideline_insufficient_evidence",
            )
        return "tool_observation_available"

    @staticmethod
    def _response_contract_error(
        response: object,
        *,
        invocation: ToolInvocation,
        definition: ToolDefinition,
    ) -> str | None:
        if not isinstance(response, AgentResponse):
            return "tool_output_not_agent_response"
        if response.request_id != invocation.request_id:
            return "tool_output_request_mismatch"
        if response.trace_id != invocation.trace_id:
            return "tool_output_trace_mismatch"
        if response.thread_id != invocation.thread_id:
            return "tool_output_thread_mismatch"
        if response.case_id != invocation.case_id:
            return "tool_output_case_mismatch"
        if response.response_kind not in definition.allowed_response_kinds:
            return "tool_output_response_kind_not_allowed"
        if response.safety_policy_id != invocation.safety_policy_id:
            return "tool_output_safety_policy_mismatch"
        return None

    @staticmethod
    def _fallback_contract_error(
        response: object,
        *,
        invocation: ToolInvocation,
    ) -> str | None:
        if not isinstance(response, AgentResponse):
            return "fallback_not_agent_response"
        if (
            response.request_id != invocation.request_id
            or response.trace_id != invocation.trace_id
            or response.thread_id != invocation.thread_id
            or response.case_id != invocation.case_id
        ):
            return "fallback_identity_mismatch"
        if response.response_kind != ResponseKind.SAFE_ABSTENTION:
            return "fallback_must_abstain"
        if response.safety_policy_id != invocation.safety_policy_id:
            return "fallback_safety_policy_mismatch"
        return None

    @staticmethod
    def _last_resort_fallback(invocation: ToolInvocation) -> AgentResponse:
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=ResponseKind.SAFE_ABSTENTION,
            summary="受控工具执行失败，系统已停止生成领域结论。",
            limitations=[
                "本系统不用于确诊或排除肺结核。",
                "请通过受控人工渠道处理；如存在持续症状或担忧，请联系医疗机构。",
            ],
            safety_policy_id=invocation.safety_policy_id,
        )
