"""Command-line interface for verified model acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..config import PROJECT_ROOT
from .manager import ArtifactError, ArtifactManager
from .manifest import ManifestError, load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bootstrap_models.py",
        description="List, plan, verify, or download SHA256-pinned TBX-Agent model files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_sources.yaml",
        help="model-source manifest (default: configs/model_sources.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="artifact cache; otherwise TBX_ARTIFACT_ROOT/platform cache is used",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "dry-run", "verify", "download"):
        child = subparsers.add_parser(command)
        child.add_argument("targets", nargs="*", help="artifact ids or group names")
        child.add_argument("--all", action="store_true", help="select every artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        manager = ArtifactManager(manifest, cache_dir=args.cache_dir)
        targets = list(args.targets)
        if args.all:
            if targets:
                parser.error("TARGET and --all are mutually exclusive")
            specs = tuple(manifest.artifacts.values())
        elif targets:
            specs = manager.select(targets)
        elif args.command == "list":
            specs = tuple(manifest.artifacts.values())
        else:
            parser.error(f"{args.command} requires at least one TARGET or --all")

        rows: list[dict[str, object]] = []
        exit_code = 0
        for spec in specs:
            if args.command == "list":
                sources = spec.resolve_sources(manager.environment)
                rows.append(
                    {
                        "artifact_id": spec.artifact_id,
                        "destination": spec.destination.as_posix(),
                        "size_bytes": spec.size_bytes,
                        "sha256": spec.sha256,
                        "revision": spec.revision,
                        "file": spec.source_file,
                        "license": spec.license.identifier,
                        "source_configured": bool(sources),
                    }
                )
            elif args.command == "dry-run":
                sources = spec.resolve_sources(manager.environment)
                status = manager.verify(spec)
                rows.append(
                    {
                        **status.to_dict(),
                        "source_configured": bool(sources),
                        "sources": [source.public_description() for source in sources],
                        "source_env_names": list(spec.source_env_names),
                        "action": "skip" if status.state == "valid" else "download",
                    }
                )
                if not sources and status.state != "valid":
                    exit_code = 2
            elif args.command == "verify":
                status = manager.verify(spec)
                rows.append(status.to_dict())
                if status.state != "valid":
                    exit_code = 1
            else:
                rows.append(manager.download(spec).to_dict())
        _emit(rows, json_output=args.json)
        return exit_code
    except (ArtifactError, ManifestError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


def _emit(rows: list[dict[str, object]], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for row in rows:
        identity = row["artifact_id"]
        state = row.get("state")
        suffix = f" [{state}]" if state else ""
        configured = row.get("source_configured")
        source_note = ""
        if configured is not None:
            source_note = " source=ready" if configured else " source=not-configured"
        print(f"{identity}: {row['destination']}{suffix}{source_note}")
