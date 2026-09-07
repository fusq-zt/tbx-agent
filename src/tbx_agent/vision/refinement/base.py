"""Backend protocol and deterministic identity for contour refinement."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from ..anatomy import AnatomyEvidence
from ..image_validator import ValidatedImage
from .models import ContourRefinementEvidence, RefinementRuntimeProbe


class RefinementBackendError(RuntimeError):
    """A configured refiner did not produce contract-valid evidence."""


class RefinementBackendUnavailable(RefinementBackendError):
    """The optional runtime or an exact verified artifact is unavailable."""


class ContourRefinementBackend(Protocol):
    backend_id: str

    def probe_runtime(self, *, load: bool = False) -> RefinementRuntimeProbe:
        ...

    def generation_key_for(
        self,
        *,
        image_sha256: str,
        boxes: list[tuple[float, float, float, float]],
        anatomy_generation_key: str,
    ) -> str:
        ...

    def refine(
        self,
        *,
        case_id: str,
        image: ValidatedImage,
        boxes: list[tuple[float, float, float, float]],
        anatomy: AnatomyEvidence,
    ) -> ContourRefinementEvidence:
        ...


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def detector_box_digest(boxes: list[tuple[float, float, float, float]]) -> str:
    return canonical_sha256(
        {
            "coordinate_space": "source_image_pixels",
            "format": "xyxy",
            "boxes": [[float(value) for value in box] for box in boxes],
        }
    )
