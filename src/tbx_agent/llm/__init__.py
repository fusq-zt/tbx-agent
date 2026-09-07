"""Constrained local-LLM runtime adapters.

The LLM is an optional presentation service. It is intentionally separated from
the deterministic routing, vision, safety, and evidence layers.
"""

from .llamacpp_client import LlamaCppClient, LlamaCppError
from .runtime_supervisor import (
    LlamaCppRuntimeConfig,
    build_server_command,
    load_runtime_config,
    verify_runtime_assets,
)
from .tool_calling import (
    HighLevelToolCall,
    HighLevelToolName,
    HighLevelToolSelection,
    NativeToolFailureCode,
    ToolSelectionMode,
    select_react_action,
)

__all__ = [
    "HighLevelToolCall",
    "HighLevelToolName",
    "HighLevelToolSelection",
    "LlamaCppClient",
    "LlamaCppError",
    "LlamaCppRuntimeConfig",
    "NativeToolFailureCode",
    "ToolSelectionMode",
    "build_server_command",
    "load_runtime_config",
    "select_react_action",
    "verify_runtime_assets",
]
