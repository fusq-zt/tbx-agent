"""Small, UI-facing client for routing-neutral anatomy visualization.

The helpers deliberately keep anatomy outside the rank03 assessment request.  A
failed or timed-out anatomy run therefore cannot replace or mutate the existing
three-class result held by the Streamlit workspace.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import requests

_TERMINAL_STATUSES = {
    "completed",
    "completed_with_refinement_failure",
    "technical_failure",
}
_ACTIVE_STATUSES = {"pending", "running"}


class AnatomyClientError(RuntimeError):
    """Stable UI boundary for transport or response-contract failures."""


@dataclass(frozen=True)
class AnatomyUIResult:
    status: str
    run: Mapping[str, Any]
    boundary_png: bytes | None = None
    contours_png: bytes | None = None


def identity_params(*, owner_scope: str, user_id: str) -> dict[str, str]:
    """Return the complete identity scope required by every anatomy endpoint."""

    if not owner_scope or not user_id:
        raise AnatomyClientError("technical_failure")
    return {"owner_scope": owner_scope, "user_id": user_id}


def _json_record(response: requests.Response, *, case_id: str) -> Mapping[str, Any]:
    if not response.ok:
        raise AnatomyClientError("technical_failure")
    try:
        payload = response.json()
    except ValueError as exc:
        raise AnatomyClientError("technical_failure") from exc
    if not isinstance(payload, Mapping):
        raise AnatomyClientError("technical_failure")
    if str(payload.get("case_id") or "") != case_id:
        raise AnatomyClientError("technical_failure")
    if str(payload.get("routing_effect") or "") != "none":
        raise AnatomyClientError("technical_failure")
    if payload.get("clinical_validation") is not False:
        raise AnatomyClientError("technical_failure")
    status = str(payload.get("status") or "")
    if status not in _ACTIVE_STATUSES | _TERMINAL_STATUSES:
        raise AnatomyClientError("technical_failure")
    if not payload.get("run_id"):
        raise AnatomyClientError("technical_failure")
    return payload


def request_and_poll_anatomy(
    *,
    api_url: str,
    case_id: str,
    owner_scope: str,
    user_id: str,
    image_name: str,
    image_bytes: bytes,
    image_mime_type: str,
    max_polls: int = 360,
    poll_interval_seconds: float = 0.5,
    request_fn: Callable[..., requests.Response] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> AnatomyUIResult:
    """Start a real anatomy run, poll it, and fetch its transparent boundary.

    The exact assessed image is uploaded again so deployments that intentionally
    do not persist source radiographs can still execute the optional tool.
    """

    if not case_id or not image_bytes or max_polls < 1 or poll_interval_seconds < 0:
        raise AnatomyClientError("technical_failure")
    requester = request_fn or requests.request
    sleeper = sleep_fn or time.sleep
    params = identity_params(owner_scope=owner_scope, user_id=user_id)
    base = api_url.rstrip("/")
    collection_path = f"/v1/cases/{case_id}/anatomy-runs"
    try:
        created_response = requester(
            "POST",
            base + collection_path,
            params=params,
            files={"file": (image_name, image_bytes, image_mime_type)},
            timeout=(5, 120),
        )
    except requests.RequestException as exc:
        raise AnatomyClientError("technical_failure") from exc
    record = _json_record(created_response, case_id=case_id)
    run_id = str(record["run_id"])

    for poll_index in range(max_polls):
        status = str(record["status"])
        if status in _TERMINAL_STATUSES:
            break
        if poll_index:
            sleeper(poll_interval_seconds)
        try:
            response = requester(
                "GET",
                base + f"{collection_path}/{run_id}",
                params=params,
                timeout=(5, 30),
            )
        except requests.RequestException as exc:
            raise AnatomyClientError("technical_failure") from exc
        record = _json_record(response, case_id=case_id)
        if str(record["run_id"]) != run_id:
            raise AnatomyClientError("technical_failure")
        if str(record["status"]) in _TERMINAL_STATUSES:
            break
    else:
        raise AnatomyClientError("technical_failure")

    if str(record["status"]) == "technical_failure":
        return AnatomyUIResult(status="technical_failure", run=record)

    try:
        boundary_response = requester(
            "GET",
            base + f"{collection_path}/{run_id}/boundary.png",
            params={**params, "structure": "combined"},
            timeout=(5, 30),
        )
    except requests.RequestException as exc:
        raise AnatomyClientError("technical_failure") from exc
    if not boundary_response.ok:
        raise AnatomyClientError("technical_failure")
    if not str(boundary_response.headers.get("Content-Type") or "").lower().startswith(
        "image/png"
    ):
        raise AnatomyClientError("technical_failure")
    routing_effect = boundary_response.headers.get("X-Anatomy-Routing-Effect")
    if routing_effect not in {None, "none"}:
        raise AnatomyClientError("technical_failure")
    boundary_png = bytes(boundary_response.content)
    if not boundary_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AnatomyClientError("technical_failure")
    contours_png = None
    if str(record.get("refinement_status") or "") == "completed":
        # A contour-layer failure degrades only this optional visualization;
        # the verified lung boundary remains usable.
        try:
            contours_response = requester(
                "GET",
                base + f"{collection_path}/{run_id}/contours.png",
                params=params,
                timeout=(5, 30),
            )
        except requests.RequestException:
            contours_response = None
        if (
            contours_response is not None
            and contours_response.ok
            and str(contours_response.headers.get("Content-Type") or "")
            .lower()
            .startswith("image/png")
            and contours_response.headers.get("X-Refinement-Routing-Effect")
            in {None, "none"}
            and bytes(contours_response.content).startswith(b"\x89PNG\r\n\x1a\n")
        ):
            contours_png = bytes(contours_response.content)
    return AnatomyUIResult(
        status=str(record["status"]),
        run=record,
        boundary_png=boundary_png,
        contours_png=contours_png,
    )
