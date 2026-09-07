"""Stable source-candidate entry point for the deterministic system benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import PROJECT_ROOT
from .system_bench import build_deterministic_candidate_configuration
from .system_bench import main as system_bench_main


def run_current_candidate(
    *,
    project_root: Path,
    output_dir: Path,
    candidate_id: str,
    config_path: Path | None = None,
) -> int:
    """Generate a source-bound candidate contract and execute it exactly once."""

    root = project_root.expanduser().resolve(strict=True)
    configuration = (
        config_path.expanduser().resolve(strict=True)
        if config_path is not None
        else root / "evaluation" / "system_bench_config.json"
    )
    configuration.relative_to(root)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    candidate_path = destination / "candidate.json"
    candidate = build_deterministic_candidate_configuration(
        root,
        evaluation_config_path=configuration,
    )
    with candidate_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(candidate, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")

    arguments = [
        "--adapter",
        "deterministic_mock",
        "--config",
        str(configuration),
        "--observations",
        str(destination / "observations.jsonl"),
        "--adapter-run-root",
        str(destination / "adapter-state"),
        "--candidate-id",
        candidate_id,
        "--candidate-config",
        str(candidate_path),
        "--output",
        str(destination / "report.json"),
        "--ledger",
        str(destination / "ledger.jsonl"),
    ]
    return system_bench_main(arguments)


def main(argv: list[str] | None = None) -> int:
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Generate the current source-bound candidate contract and run the synthetic "
            "system release gates."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts" / "system-bench",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "evaluation" / "system_bench_config.json",
        help=(
            "Versioned system-bench configuration. The default is current v1.6; "
            "use an explicitly versioned configuration for historical replay."
        ),
    )
    parser.add_argument("--candidate-id", default="current-source-candidate")
    args = parser.parse_args(argv)
    return run_current_candidate(
        project_root=project_root,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
