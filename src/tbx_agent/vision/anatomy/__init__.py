"""Optional anatomy segmentation and detector-to-lung-field localization."""

from .base import (
    AnatomyBackend,
    AnatomyBackendError,
    AnatomyBackendUnavailable,
    build_generation_key,
)
from .localization import (
    LungFieldLocalizationPolicy,
    localize_bbox_to_lung_fields,
    localize_detection_boxes,
)
from .models import (
    AnatomyEvidence,
    AnatomyMask,
    AnatomyQCReport,
    AnatomyQCStatus,
    AnatomyRuntimeProbe,
    AnatomySpatialSummary,
    CompactRLE,
    DetectionAnatomyLocation,
    LungFieldAssignment,
    LungFieldZone,
    LungSide,
)
from .presentation import (
    SPATIAL_PRESENTATION_POLICY_ID,
    build_chat_spatial_summary,
    build_spatial_summary,
)
from .qc import AnatomyQCPolicy, evaluate_lung_masks
from .rle import MaskEncodingError, decode_binary_mask, encode_binary_mask
from .xrv_pspnet import TorchXRayVisionPSPNetBackend, XRVPSPNetConfig

__all__ = [
    "AnatomyBackend",
    "AnatomyBackendError",
    "AnatomyBackendUnavailable",
    "AnatomyEvidence",
    "AnatomyMask",
    "AnatomyQCPolicy",
    "AnatomyQCReport",
    "AnatomyQCStatus",
    "AnatomyRuntimeProbe",
    "AnatomySpatialSummary",
    "CompactRLE",
    "DetectionAnatomyLocation",
    "LungFieldAssignment",
    "LungFieldLocalizationPolicy",
    "LungFieldZone",
    "LungSide",
    "MaskEncodingError",
    "TorchXRayVisionPSPNetBackend",
    "XRVPSPNetConfig",
    "build_generation_key",
    "build_chat_spatial_summary",
    "build_spatial_summary",
    "decode_binary_mask",
    "encode_binary_mask",
    "evaluate_lung_masks",
    "localize_bbox_to_lung_fields",
    "localize_detection_boxes",
    "SPATIAL_PRESENTATION_POLICY_ID",
]
