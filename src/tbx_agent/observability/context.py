from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REQUEST_ID: ContextVar[str | None] = ContextVar("tbx_request_id", default=None)


def bind_request_id(candidate: str | None) -> tuple[str, Token[str | None]]:
    request_id = (
        candidate
        if candidate is not None and _REQUEST_ID_PATTERN.fullmatch(candidate)
        else uuid.uuid4().hex
    )
    return request_id, _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    _REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    return _REQUEST_ID.get()
