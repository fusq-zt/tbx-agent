#!/usr/bin/env sh
set -eu
umask 077

# Build an auditable *candidate* bundle from the pinned llama.cpp b10517 source.
# This command never registers its own output: registration is a separate,
# explicit review boundary with an independently checked binary digest.

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-"$PROJECT_ROOT/.venv/bin/python"}
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi

if [ -n "${TBX_RUNTIME_ROOT:-}" ]; then
    RUNTIME_ROOT=$TBX_RUNTIME_ROOT
elif [ -n "${TBX_AGENT_DATA_ROOT:-}" ]; then
    RUNTIME_ROOT=$TBX_AGENT_DATA_ROOT
elif [ -n "${XDG_DATA_HOME:-}" ]; then
    RUNTIME_ROOT=$XDG_DATA_HOME/tbx-agent
elif [ -n "${HOME:-}" ]; then
    RUNTIME_ROOT=$HOME/.local/share/tbx-agent
else
    echo "Set --runtime-root when HOME and XDG_DATA_HOME are unavailable." >&2
    exit 2
fi

BACKEND=cpu
BUILD_DIR=""
BUNDLE_DIR=""
JOBS=""
EXECUTE=0

usage() {
    cat <<'EOF'
Usage: scripts/build_llamacpp_linux.sh [options]
  --backend cpu|cuda    ggml backend (default: cpu)
  --runtime-root PATH   external downloads/build/receipt root
  --build-dir PATH      external CMake build directory
  --bundle-dir PATH     new immutable candidate bundle directory
  --jobs N              parallel CMake jobs (default: online CPUs)
  --python PATH         project Python executable
  --execute             perform downloads and compilation (default is dry-run)
  --dry-run             print the plan without changing files (default)

The output is an unregistered candidate. Review its receipt and verify the
reported llama-server SHA-256 independently before running the printed
register-runtime command.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --backend|--runtime-root|--build-dir|--bundle-dir|--jobs|--python)
            option=$1
            [ "$#" -ge 2 ] || { echo "$option requires a value." >&2; exit 2; }
            case "$option" in
                --backend) BACKEND=$2 ;;
                --runtime-root) RUNTIME_ROOT=$2 ;;
                --build-dir) BUILD_DIR=$2 ;;
                --bundle-dir) BUNDLE_DIR=$2 ;;
                --jobs) JOBS=$2 ;;
                --python) PYTHON=$2 ;;
            esac
            shift 2
            ;;
        --execute) EXECUTE=1; shift ;;
        --dry-run) EXECUTE=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$BACKEND" in cpu|cuda) ;; *) echo "--backend must be cpu or cuda." >&2; exit 2 ;; esac
if [ -z "$JOBS" ]; then
    JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
fi
case "$JOBS" in ''|*[!0-9]*) echo "--jobs must be an integer from 1 to 256." >&2; exit 2 ;; esac
[ "$JOBS" -ge 1 ] && [ "$JOBS" -le 256 ] || {
    echo "--jobs must be an integer from 1 to 256." >&2
    exit 2
}

ARCH=$(uname -m 2>/dev/null || printf 'unknown')
case "$ARCH" in
    x86_64|amd64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=aarch64 ;;
    *) echo "Unsupported Linux architecture: $ARCH" >&2; exit 2 ;;
esac

[ -n "$BUILD_DIR" ] || BUILD_DIR=$RUNTIME_ROOT/build/llama.cpp-b10517-$BACKEND-$ARCH
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -n "$BUNDLE_DIR" ] || \
    BUNDLE_DIR=$RUNTIME_ROOT/candidates/llama.cpp-b10517-$BACKEND-$ARCH-$STAMP
SOURCE_DIR=$RUNTIME_ROOT/sources/llama.cpp-b10517
SOURCE_ARCHIVE=$RUNTIME_ROOT/downloads/llama.cpp/b10517/llama.cpp-b10517-source.zip
PLATFORM_ID=linux-$BACKEND-$ARCH

if [ "$EXECUTE" -eq 0 ]; then
    echo "DRY RUN: no file, environment, source checkout, or model is changed."
    echo "Pinned source: ggml-org/llama.cpp b10517 (commit dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe)"
    echo "Backend: $BACKEND"
    echo "Runtime root: $RUNTIME_ROOT"
    echo "Build directory: $BUILD_DIR"
    echo "Candidate bundle: $BUNDLE_DIR"
    echo "Parallel jobs: $JOBS"
    echo "Execution requires Linux, CMake, a C++ compiler, network access for the pinned source,"
    echo "and --execute. CUDA additionally requires a compatible CUDA toolkit and driver."
    exit 0
fi

[ "$(uname -s)" = Linux ] || { echo "This builder runs only on Linux." >&2; exit 2; }
command -v "$PYTHON" >/dev/null 2>&1 || [ -x "$PYTHON" ] || {
    echo "Python executable not found: $PYTHON" >&2
    exit 2
}
command -v cmake >/dev/null 2>&1 || { echo "CMake is required." >&2; exit 2; }
if [ "$BACKEND" = cuda ] && ! command -v nvcc >/dev/null 2>&1; then
    if [ -z "${CUDACXX:-}" ] || [ ! -x "$CUDACXX" ]; then
        echo "CUDA build requires nvcc on PATH or an executable CUDACXX." >&2
        exit 2
    fi
fi

# Resolve before the containment check so ../ and symlink aliases cannot place
# large build outputs back inside the Git checkout.
RUNTIME_ROOT=$(
    "$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
        "$RUNTIME_ROOT"
)
BUILD_DIR=$(
    "$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
        "$BUILD_DIR"
)
BUNDLE_DIR=$(
    "$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
        "$BUNDLE_DIR"
)
[ "$RUNTIME_ROOT" != / ] || { echo "Runtime root cannot be the filesystem root." >&2; exit 2; }
case "$RUNTIME_ROOT/" in "$PROJECT_ROOT/"*) echo "Runtime root must be outside Git." >&2; exit 2 ;; esac
case "$BUILD_DIR" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Build directory must be a child of the runtime root." >&2; exit 2 ;;
esac
case "$BUNDLE_DIR" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Bundle directory must be a child of the runtime root." >&2; exit 2 ;;
esac
case "$BUILD_DIR/" in "$PROJECT_ROOT/"*) echo "Build directory must be outside Git." >&2; exit 2 ;; esac
case "$BUNDLE_DIR/" in "$PROJECT_ROOT/"*) echo "Bundle directory must be outside Git." >&2; exit 2 ;; esac
[ ! -e "$BUNDLE_DIR" ] || { echo "Candidate bundle already exists: $BUNDLE_DIR" >&2; exit 2; }

SOURCE_DIR=$RUNTIME_ROOT/sources/llama.cpp-b10517
SOURCE_ARCHIVE=$RUNTIME_ROOT/downloads/llama.cpp/b10517/llama.cpp-b10517-source.zip
"$PYTHON" "$PROJECT_ROOT/scripts/bootstrap_qwen.py" \
    --runtime-root "$RUNTIME_ROOT" download-llama-source
[ -f "$SOURCE_DIR/CMakeLists.txt" ] || { echo "Pinned source is incomplete." >&2; exit 2; }
[ -f "$SOURCE_ARCHIVE" ] || { echo "Pinned source archive is missing." >&2; exit 2; }

CUDA_FLAG=OFF
[ "$BACKEND" = cuda ] && CUDA_FLAG=ON
mkdir -p "$BUILD_DIR"
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA="$CUDA_FLAG"
cmake --build "$BUILD_DIR" --config Release \
    --target llama-server llama-quantize --parallel "$JOBS"

SERVER=$BUILD_DIR/bin/llama-server
QUANTIZE=$BUILD_DIR/bin/llama-quantize
[ -x "$SERVER" ] || { echo "CMake did not produce executable $SERVER" >&2; exit 2; }
[ -x "$QUANTIZE" ] || { echo "CMake did not produce executable $QUANTIZE" >&2; exit 2; }

TEMP_BUNDLE=$BUNDLE_DIR.candidate.$$
cleanup() { [ ! -e "$TEMP_BUNDLE" ] || rm -rf -- "$TEMP_BUNDLE"; }
trap cleanup EXIT INT TERM HUP
mkdir -p "$(dirname -- "$BUNDLE_DIR")" "$TEMP_BUNDLE"
cp "$SERVER" "$TEMP_BUNDLE/llama-server"
cp "$QUANTIZE" "$TEMP_BUNDLE/llama-quantize"
chmod 0755 "$TEMP_BUNDLE/llama-server" "$TEMP_BUNDLE/llama-quantize"
mv "$TEMP_BUNDLE" "$BUNDLE_DIR"
trap - EXIT INT TERM HUP

RECEIPT=$RUNTIME_ROOT/provenance/llama-b10517-$BACKEND-$ARCH-$STAMP-candidate.json
mkdir -p "$(dirname -- "$RECEIPT")"
"$PYTHON" - "$PROJECT_ROOT" "$RUNTIME_ROOT" "$BUNDLE_DIR" "$SOURCE_ARCHIVE" "$RECEIPT" \
    "$BACKEND" "$PLATFORM_ID" "$BUILD_DIR" "$JOBS" <<'PY'
from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


project, runtime_root, bundle, source_archive, receipt, backend, platform_id, build_dir, jobs = (
    sys.argv[1:]
)
project_path = Path(project)
bundle_path = Path(bundle)
source_path = Path(source_archive)
receipt_path = Path(receipt)
cache_path = Path(build_dir) / "CMakeCache.txt"
if not cache_path.is_file():
    raise SystemExit("CMakeCache.txt is missing from the completed build")
contract = yaml.safe_load(
    (project_path / "configs/qwen_runtime_sources.yaml").read_text(encoding="utf-8")
)
source_pin = contract["llama_cpp"]["source_archive"]
source_digest = sha256(source_path)
if source_digest != source_pin["sha256"] or source_path.stat().st_size != source_pin["size_bytes"]:
    raise SystemExit("pinned llama.cpp source archive changed after verification")
files = []
for path in sorted(bundle_path.iterdir(), key=lambda item: item.name):
    if path.is_symlink() or not path.is_file():
        raise SystemExit("candidate bundle contains a non-regular file")
    files.append(
        {"relative_path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    )
binary_digest = next(item["sha256"] for item in files if item["relative_path"] == "llama-server")
argv = [
    sys.executable,
    str(project_path / "scripts/bootstrap_qwen.py"),
    "--runtime-root",
    str(Path(runtime_root)),
    "register-runtime",
    "--bundle-dir",
    str(bundle_path),
    "--binary-name",
    "llama-server",
    "--expected-binary-sha256",
    binary_digest,
    "--release-archive",
    str(source_path),
    "--expected-release-sha256",
    source_pin["sha256"],
    "--platform-id",
    platform_id,
]
payload = {
    "schema_version": 1,
    "state": "candidate_unregistered",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "backend": backend,
    "platform_id": platform_id,
    "build_dir": str(Path(build_dir)),
    "parallel_jobs": int(jobs),
    "source": {
        "project": contract["llama_cpp"]["project"],
        "tag": contract["llama_cpp"]["tag"],
        "commit": contract["llama_cpp"]["commit"],
        "archive_path": str(source_path),
        "archive_size_bytes": source_path.stat().st_size,
        "archive_sha256": source_digest,
    },
    "cmake_contract": {
        "build_type": "Release",
        "build_shared_libs": False,
        "llama_curl": False,
        "ggml_native": False,
        "ggml_cuda": backend == "cuda",
        "cmake_version": subprocess.check_output(
            ["cmake", "--version"], text=True, encoding="utf-8"
        ).splitlines()[0],
        "cmake_cache_sha256": sha256(cache_path),
    },
    "build_platform": {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "bundle_dir": str(bundle_path),
    "files": files,
    "observed_binary_sha256": binary_digest,
    "registration_argv_after_independent_digest_review": argv,
}
receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Candidate bundle: {bundle_path}")
print(f"Candidate receipt: {receipt_path}")
print(f"Observed llama-server SHA-256: {binary_digest}")
print("Review that digest independently before registration. Suggested command:")
print(shlex.join(argv))
PY

echo "Candidate only: no runtime identity or generated LLM configuration was changed."
