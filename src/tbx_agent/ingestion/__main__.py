from __future__ import annotations

import argparse
import json
from pathlib import Path

from .errors import IngestionError
from .pipeline import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline TBX guideline snapshot")
    parser.add_argument("--config", type=Path, required=True, help="strict YAML ingestion config")
    args = parser.parse_args()
    try:
        result = run_ingestion(args.config)
    except IngestionError as exc:
        parser.exit(2, f"ingestion failed: {exc}\n")
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "build_id": result.build_id,
                "manifest": str(result.manifest_path),
                "chunks": str(result.chunks_path),
                "receipt": str(result.receipt_path),
                "chunk_count": result.chunk_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
