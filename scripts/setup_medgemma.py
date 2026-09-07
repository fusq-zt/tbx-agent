"""Repository-local entry point for hash-verified MedGemma setup."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if __name__ == "__main__":
    main = import_module("tbx_agent.llm.medgemma_setup").main
    raise SystemExit(main())
