from __future__ import annotations

from typing import Protocol

from ..schemas import DetectionEvidence, VisionEvidence
from .image_validator import ValidatedImage


class VisionBackendError(RuntimeError):
    """Raised when a version-pinned vision backend cannot produce valid evidence."""


class VisionBackend(Protocol):
    backend_id: str

    def infer(self, *, case_id: str, image: ValidatedImage) -> VisionEvidence:
        """Run the screening classifier and return structured evidence."""

    def localize(
        self, *, case_id: str, image: ValidatedImage
    ) -> list[DetectionEvidence]:
        """Run the optional localization worker for the same immutable image."""
