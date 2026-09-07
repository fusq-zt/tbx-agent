"""Backend contract and cache-key construction for anatomy evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from ..image_validator import ValidatedImage
from .models import AnatomyEvidence, AnatomyRuntimeProbe


class AnatomyBackendError(RuntimeError):
    """An anatomy backend was configured but did not produce trustworthy masks."""


class AnatomyBackendUnavailable(AnatomyBackendError):
    """Optional runtime or pinned model artifacts are unavailable."""


class AnatomyBackend(Protocol):
    backend_id: str

    @property
    def loaded(self) -> bool: ...

    def probe_runtime(self, *, load: bool = False) -> AnatomyRuntimeProbe:
        """Inspect optional-runtime state, loading weights only when requested."""

    def generation_key_for(self, image_sha256: str) -> str:
        """Build the cache key for an image, loading weight identity if necessary."""

    def infer(self, *, case_id: str, image: ValidatedImage) -> AnatomyEvidence:
        """Return independent source-coordinate anatomy evidence.

        Implementations must not read or mutate the rank03 routing policy.
        """


def build_generation_key(
    *,
    image_sha256: str,
    model_weight_sha256: str,
    preprocessing_id: str,
    policy_id: str,
    backend_id: str,
    parameters: dict[str, object] | None = None,
) -> str:
    """Hash every input that can change an anatomy artifact."""

    for name, value in (
        ("image_sha256", image_sha256),
        ("model_weight_sha256", model_weight_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    payload = {
        "backend_id": backend_id,
        "image_sha256": image_sha256,
        "model_weight_sha256": model_weight_sha256,
        "parameters": parameters or {},
        "policy_id": policy_id,
        "preprocessing_id": preprocessing_id,
        "schema": "tbx-anatomy-generation-key-v1",
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
