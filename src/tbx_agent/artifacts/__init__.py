"""Verified, manifest-driven acquisition of large model artifacts."""

from .manager import (
    ArtifactManager,
    ArtifactStatus,
    DownloadError,
    FileLockTimeoutError,
    MissingSourceError,
    default_artifact_root,
    sha256_file,
)
from .manifest import (
    ArtifactManifest,
    ArtifactSpec,
    LicenseMetadata,
    ManifestError,
    SourceSpec,
    load_manifest,
)
from .source_checkout import (
    DFINE_REPOSITORY,
    DFINE_REVISION,
    SourceCheckoutError,
    SourceCheckoutReceipt,
    acquire_dfine_source,
    verify_dfine_source,
)

__all__ = [
    "DFINE_REPOSITORY",
    "DFINE_REVISION",
    "ArtifactManager",
    "ArtifactManifest",
    "ArtifactSpec",
    "ArtifactStatus",
    "DownloadError",
    "FileLockTimeoutError",
    "LicenseMetadata",
    "ManifestError",
    "MissingSourceError",
    "SourceCheckoutError",
    "SourceCheckoutReceipt",
    "SourceSpec",
    "acquire_dfine_source",
    "default_artifact_root",
    "load_manifest",
    "sha256_file",
    "verify_dfine_source",
]
