from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from .openai_compat import OPENAI_COMPAT_SMOKE_ENDPOINTS, OPENAI_COMPATIBLE_ENDPOINTS
from .runtime_supervisor import (
    build_server_command,
    load_runtime_config,
    verify_runtime_assets,
)


def _api_key(path: Path) -> str:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values[0]) < 32:
        raise ValueError("llama.cpp API key file is invalid")
    return values[0]


def _probe(config, *, timeout: float) -> dict[str, object]:
    base = f"http://{config.host}:{config.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"Authorization": f"Bearer {_api_key(config.api_key_file)}"}

    def request(path: str) -> object:
        with opener.open(
            urllib.request.Request(base + path, headers=headers), timeout=timeout
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"llama.cpp probe returned HTTP {response.status}")
            return json.loads(response.read(1024 * 1024))

    health = request("/health")
    models = request("/v1/models")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError("llama.cpp health probe is not ready")
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise RuntimeError("llama.cpp model inventory response is invalid")
    aliases = {
        row.get("id") for row in models["data"] if isinstance(row, dict) and row.get("id")
    }
    if config.model_alias not in aliases:
        raise RuntimeError("llama.cpp is not serving the configured model alias")
    return {"status": "ok", "base_url": base, "model_alias": config.model_alias}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a pinned llama.cpp runtime contract")
    parser.add_argument("action", choices=("verify", "argv", "info", "probe", "serve"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="probe only after this invocation's caller already verified the pinned assets",
    )
    args = parser.parse_args()

    config = load_runtime_config(args.config.resolve())
    if args.action == "info":
        payload = {
            "runtime_id": config.runtime_id,
            "base_url": f"http://{config.host}:{config.port}",
            "openai_base_url": f"http://{config.host}:{config.port}/v1",
            "openai_compatible_endpoints": list(OPENAI_COMPATIBLE_ENDPOINTS),
            "openai_compat_smoke_endpoints": list(OPENAI_COMPAT_SMOKE_ENDPOINTS),
            "model_alias": config.model_alias,
            "model_path": str(config.model_path),
            "api_key_file": str(config.api_key_file),
            "server_build": config.server_build,
            "runtime_config_sha256": config.canonical_sha256(),
        }
    elif args.action == "probe":
        if not args.skip_assets:
            verify_runtime_assets(config)
        payload = _probe(config, timeout=args.timeout)
    elif args.action == "serve":
        verify_runtime_assets(config)
        command = build_server_command(config)
        os.execv(command[0], command)
        raise RuntimeError("failed to replace the supervisor process")  # pragma: no cover
    else:
        attestation = verify_runtime_assets(config)
        payload = (
            attestation
            if args.action == "verify"
            else {"argv": build_server_command(config), "attestation": attestation}
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
