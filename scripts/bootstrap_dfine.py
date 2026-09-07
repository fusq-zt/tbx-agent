#!/usr/bin/env python3
"""Install and verify the pinned D-FINE source outside this repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tbx_agent.artifacts import default_artifact_root  # noqa: E402
from tbx_agent.artifacts.source_checkout import (  # noqa: E402
    SourceCheckoutError,
    acquire_dfine_source,
    verify_dfine_source,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_artifact_root())
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify_only:
            receipt = verify_dfine_source(arguments.cache_dir)
        else:
            receipt = acquire_dfine_source(arguments.cache_dir)
    except SourceCheckoutError as exc:
        print(f"D-FINE bootstrap failed: {exc}", file=sys.stderr)
        return 2
    payload = receipt.to_dict()
    if arguments.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"D-FINE {payload['state']}: {payload['destination']} @ {payload['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
