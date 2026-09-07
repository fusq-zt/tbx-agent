"""Optional detector-box contour refinement backends."""

from .base import (
    ContourRefinementBackend,
    RefinementBackendError,
    RefinementBackendUnavailable,
    canonical_sha256,
    detector_box_digest,
)
from .medsam_hf import (
    HFMedSAMBoxRefinementBackend,
    HFMedSAMConfig,
    MedSAMRefinementPolicy,
)
from .models import (
    ContourRefinementEvidence,
    DetectionContourEvidence,
    DetectionRefinementStatus,
    RefinementRuntimeProbe,
)

__all__ = [
    "ContourRefinementBackend",
    "ContourRefinementEvidence",
    "DetectionContourEvidence",
    "DetectionRefinementStatus",
    "HFMedSAMBoxRefinementBackend",
    "HFMedSAMConfig",
    "MedSAMRefinementPolicy",
    "RefinementBackendError",
    "RefinementBackendUnavailable",
    "RefinementRuntimeProbe",
    "canonical_sha256",
    "detector_box_digest",
]
