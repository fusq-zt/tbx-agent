from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import requests
from PIL import Image

from ui.anatomy_client import AnatomyClientError, request_and_poll_anatomy


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 6), (0, 0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _response(
    payload: Mapping[str, Any] | None = None,
    *,
    content: bytes = b"",
    content_type: str = "application/json",
    status_code: int = 200,
    headers: Mapping[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers["Content-Type"] = content_type
    response.headers.update(headers or {})
    response._content = (
        json.dumps(payload).encode("utf-8") if payload is not None else content
    )
    return response


def _run(
    status: str,
    *,
    refinement_status: str | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": "anatomy-run-1",
        "case_id": "case-1",
        "status": status,
        "routing_effect": "none",
        "clinical_validation": False,
    }
    if refinement_status is not None:
        payload["refinement_status"] = refinement_status
    return payload


@dataclass(frozen=True)
class _Call:
    method: str
    url: str
    kwargs: Mapping[str, Any]


def test_real_anatomy_flow_reuploads_exact_image_and_scopes_every_request() -> None:
    calls: list[_Call] = []
    records = iter([_run("pending"), _run("running"), _run("completed")])

    def request(method: str, url: str, **kwargs) -> requests.Response:
        calls.append(_Call(method=method, url=url, kwargs=kwargs))
        if url.endswith("boundary.png"):
            return _response(
                content=_png_bytes(),
                content_type="image/png",
                headers={"X-Anatomy-Routing-Effect": "none"},
            )
        return _response(next(records), status_code=202 if method == "POST" else 200)

    image_bytes = b"exact-assessed-image"
    result = request_and_poll_anatomy(
        api_url="http://api.test/",
        case_id="case-1",
        owner_scope="tenant:one",
        user_id="patient-1",
        image_name="cxr.png",
        image_bytes=image_bytes,
        image_mime_type="image/png",
        max_polls=3,
        poll_interval_seconds=0,
        request_fn=request,
        sleep_fn=lambda _seconds: None,
    )

    assert result.status == "completed"
    assert result.boundary_png == _png_bytes()
    assert [call.method for call in calls] == ["POST", "GET", "GET", "GET"]
    expected_identity = {"owner_scope": "tenant:one", "user_id": "patient-1"}
    for call in calls[:-1]:
        assert call.kwargs["params"] == expected_identity
    assert calls[-1].kwargs["params"] == {**expected_identity, "structure": "combined"}
    assert calls[0].kwargs["files"] == {
        "file": ("cxr.png", image_bytes, "image/png")
    }


def test_technical_failure_is_terminal_and_does_not_fetch_boundary() -> None:
    calls: list[str] = []

    def request(method: str, url: str, **_kwargs) -> requests.Response:
        calls.append(f"{method} {url}")
        return _response(_run("technical_failure"), status_code=202)

    result = request_and_poll_anatomy(
        api_url="http://api.test",
        case_id="case-1",
        owner_scope="tenant:one",
        user_id="patient-1",
        image_name="cxr.png",
        image_bytes=b"image",
        image_mime_type="image/png",
        request_fn=request,
    )

    assert result.status == "technical_failure"
    assert result.boundary_png is None
    assert len(calls) == 1


def test_response_cannot_claim_anatomy_changed_routing() -> None:
    unsafe = {**_run("completed"), "routing_effect": "changed"}

    with pytest.raises(AnatomyClientError, match="technical_failure"):
        request_and_poll_anatomy(
            api_url="http://api.test",
            case_id="case-1",
            owner_scope="tenant:one",
            user_id="patient-1",
            image_name="cxr.png",
            image_bytes=b"image",
            image_mime_type="image/png",
            request_fn=lambda *_args, **_kwargs: _response(unsafe),
        )


def test_completed_refinement_fetches_verified_contour_layer() -> None:
    calls: list[str] = []

    def request(method: str, url: str, **_kwargs) -> requests.Response:
        calls.append(f"{method} {url}")
        if url.endswith("boundary.png"):
            return _response(
                content=_png_bytes(),
                content_type="image/png",
                headers={"X-Anatomy-Routing-Effect": "none"},
            )
        if url.endswith("contours.png"):
            return _response(
                content=_png_bytes(),
                content_type="image/png",
                headers={
                    "X-Refinement-Routing-Effect": "none",
                    "X-Clinical-Validation": "false",
                },
            )
        return _response(_run("completed", refinement_status="completed"), status_code=202)

    result = request_and_poll_anatomy(
        api_url="http://api.test",
        case_id="case-1",
        owner_scope="tenant:one",
        user_id="patient-1",
        image_name="cxr.png",
        image_bytes=b"image",
        image_mime_type="image/png",
        request_fn=request,
    )

    assert result.status == "completed"
    assert result.boundary_png == _png_bytes()
    assert result.contours_png == _png_bytes()
    assert [call.rsplit("/", 1)[-1] for call in calls] == [
        "anatomy-runs",
        "boundary.png",
        "contours.png",
    ]


def test_refinement_failure_preserves_completed_lung_boundary() -> None:
    calls: list[str] = []

    def request(method: str, url: str, **_kwargs) -> requests.Response:
        calls.append(f"{method} {url}")
        if url.endswith("boundary.png"):
            return _response(
                content=_png_bytes(),
                content_type="image/png",
                headers={"X-Anatomy-Routing-Effect": "none"},
            )
        return _response(
            _run(
                "completed_with_refinement_failure",
                refinement_status="technical_failure",
            ),
            status_code=202,
        )

    result = request_and_poll_anatomy(
        api_url="http://api.test",
        case_id="case-1",
        owner_scope="tenant:one",
        user_id="patient-1",
        image_name="cxr.png",
        image_bytes=b"image",
        image_mime_type="image/png",
        request_fn=request,
    )

    assert result.status == "completed_with_refinement_failure"
    assert result.boundary_png == _png_bytes()
    assert result.contours_png is None
    assert not any(call.endswith("contours.png") for call in calls)


def test_invalid_contour_response_degrades_only_contour_layer() -> None:
    def request(method: str, url: str, **_kwargs) -> requests.Response:
        if url.endswith("boundary.png"):
            return _response(
                content=_png_bytes(),
                content_type="image/png",
                headers={"X-Anatomy-Routing-Effect": "none"},
            )
        if url.endswith("contours.png"):
            return _response(
                content=b"not-a-png",
                content_type="image/png",
                headers={"X-Refinement-Routing-Effect": "none"},
            )
        return _response(_run("completed", refinement_status="completed"), status_code=202)

    result = request_and_poll_anatomy(
        api_url="http://api.test",
        case_id="case-1",
        owner_scope="tenant:one",
        user_id="patient-1",
        image_name="cxr.png",
        image_bytes=b"image",
        image_mime_type="image/png",
        request_fn=request,
    )

    assert result.status == "completed"
    assert result.boundary_png == _png_bytes()
    assert result.contours_png is None
