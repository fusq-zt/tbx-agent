"""Repository-local entry point for the external Qwen/llama.cpp bootstrap."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return import_module("tbx_agent.llm.bootstrap_cli").main()


if __name__ == "__main__":
    raise SystemExit(main())
