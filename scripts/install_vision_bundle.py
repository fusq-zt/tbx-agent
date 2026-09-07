"""Install an independently acquired rank03 inference bundle outside this checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbx_agent.artifacts.rank03_bundle import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
