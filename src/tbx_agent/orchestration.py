from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .langgraph_runtime import compiled_graph, workflow
from .routing import route_tool as route_tool
from .schemas import AgentResponse
from .service import TBXAgentService
from .tools.contracts import ToolStatus

if TYPE_CHECKING:
    from .agent_runtime import AgentTurnResult


class AgentTurnRequest(BaseModel):
    """Strict public boundary for one synchronous agent turn."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=4_000)
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    owner_scope: str = Field(min_length=1, max_length=256)
    case_id: str | None = Field(default=None, min_length=1, max_length=256)


@dataclass(slots=True)
class _LockEntry:
    lock: RLock
    users: int = 0


class _ThreadLockPool:
    """Serialize turns only within the same tenant/user/thread partition."""

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, _LockEntry] = {}

    @contextmanager
    def hold(self, key: str):
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _LockEntry(lock=RLock())
                self._entries[key] = entry
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(key, None)


def tenant_thread_namespace(*, owner_scope: str, user_id: str, thread_id: str) -> str:
    """Return an opaque key for the in-process tenant thread lock."""

    values = (owner_scope.strip(), user_id.strip(), thread_id.strip())
    if not all(values):
        raise ValueError("thread identity fields must be non-empty")
    material = "\0".join(values).encode("utf-8")
    return f"tbx-{hashlib.sha256(material).hexdigest()}"


def checkpoint_namespace(*, owner_scope: str, user_id: str, thread_id: str) -> str:
    """Backward-compatible alias; no checkpoint is created or persisted."""

    return tenant_thread_namespace(
        owner_scope=owner_scope,
        user_id=user_id,
        thread_id=thread_id,
    )


class TBXAgentGraph:
    """Synchronous API boundary backed by the compiled Plan + ReAct StateGraph."""

    def __init__(self, service: TBXAgentService):
        self.service = service
        self.workflow = workflow
        self.compiled_graph = compiled_graph
        self._invoke_locks = _ThreadLockPool()

    def invoke(self, request: dict[str, Any]) -> AgentResponse:
        return self.invoke_with_receipt(request).response

    def invoke_with_receipt(
        self,
        request: dict[str, Any],
        *,
        narrator_override: Any | None = None,
    ) -> AgentTurnResult:
        # Validate the caller's complete payload. Internal execution fields are
        # not filtered or trusted at this boundary.
        validated = AgentTurnRequest.model_validate(request)
        namespace = tenant_thread_namespace(
            owner_scope=validated.owner_scope,
            user_id=validated.user_id,
            thread_id=validated.thread_id,
        )
        with self._invoke_locks.hold(namespace):
            generator = (
                narrator_override
                if narrator_override is not None
                else self.service.narrator
            )
            return self.service.respond_with_controller(
                message=validated.message,
                thread_id=validated.thread_id,
                user_id=validated.user_id,
                owner_scope=validated.owner_scope,
                case_id=validated.case_id,
                generator=generator,
                narrator_override=narrator_override,
            )

    def tool_statuses(self) -> list[ToolStatus]:
        return self.service.tool_registry.statuses()

    def close(self) -> None:
        """Compatibility no-op; the graph deliberately has no checkpointer."""

        return None
