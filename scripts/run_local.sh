#!/usr/bin/env sh
set -eu
umask 077

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

load_dotenv() {
    file=$1
    [ -f "$file" ] || return 0
    while IFS= read -r raw || [ -n "$raw" ]; do
        line=$(printf '%s' "$raw" | tr -d '\r')
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*}
        value=${line#*=}
        if [ "$key" = "$line" ] || ! printf '%s' "$key" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*$'; then
            echo "Invalid .env line; shell expansion is intentionally unsupported: $raw" >&2
            exit 2
        fi
        case "$value" in
            \"*\") value=${value#\"}; value=${value%\"} ;;
            \'*\') value=${value#\'}; value=${value%\'} ;;
            \"*|\'*)
                echo "Invalid .env line; quoted values must have a closing quote: $key" >&2
                exit 2
                ;;
        esac
        eval "existing=\${$key-}"
        [ -z "$existing" ] || continue
        export "$key=$value"
    done < "$file"
}

truthy() {
    case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac
}

platform_artifact_root() {
    if [ -n "${XDG_CACHE_HOME:-}" ]; then
        printf '%s\n' "$XDG_CACHE_HOME/tbx-agent/artifacts"
    elif [ -n "${HOME:-}" ]; then
        printf '%s\n' "$HOME/.cache/tbx-agent/artifacts"
    else
        echo "Set TBX_ARTIFACT_ROOT when HOME and XDG_CACHE_HOME are unavailable." >&2
        exit 2
    fi
}

load_dotenv "$PROJECT_ROOT/.env"
export TBX_AGENT_PROJECT_ROOT="$PROJECT_ROOT"

PYTHON=${PYTHON:-"$PROJECT_ROOT/.venv/bin/python"}
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi

if [ -z "${TBX_AGENT_DATA_ROOT:-}" ]; then
    if [ -n "${TBX_RUNTIME_ROOT:-}" ]; then
        TBX_AGENT_DATA_ROOT=$TBX_RUNTIME_ROOT
    elif [ -n "${XDG_DATA_HOME:-}" ]; then
        TBX_AGENT_DATA_ROOT=$XDG_DATA_HOME/tbx-agent
    elif [ -n "${HOME:-}" ]; then
        TBX_AGENT_DATA_ROOT=$HOME/.local/share/tbx-agent
    else
        echo "Set TBX_AGENT_DATA_ROOT when HOME and XDG_DATA_HOME are unavailable." >&2
        exit 2
    fi
    export TBX_AGENT_DATA_ROOT
fi
LLM_ENV_FILE=${TBX_AGENT_LLM_ENV_FILE:-"$TBX_AGENT_DATA_ROOT/config/llm.env"}
load_dotenv "$LLM_ENV_FILE"
export TBX_AGENT_DB_PATH=${TBX_AGENT_DB_PATH:-"$TBX_AGENT_DATA_ROOT/tbx_agent.sqlite3"}
if [ -z "${TBX_ARTIFACT_ROOT:-}" ]; then
    # Match bootstrap.sh/default_artifact_root. An explicit process variable or
    # .env entry always wins; never redirect models into mutable runtime data.
    TBX_ARTIFACT_ROOT=$(platform_artifact_root)
    export TBX_ARTIFACT_ROOT
fi

BIND_HOST=${TBX_AGENT_BIND_HOST:-127.0.0.1}
BIND_PORT=${TBX_AGENT_BIND_PORT:-8000}
UI_PORT=${TBX_AGENT_UI_PORT:-8501}
API_URL="http://127.0.0.1:$BIND_PORT"
export TBX_AGENT_API_URL="$API_URL"
VISION_BACKEND=${TBX_AGENT_VISION_BACKEND:-rank03}
NARRATOR_BACKEND=${TBX_AGENT_NARRATOR_BACKEND:-llama_cpp}
REQUIRE_VISION=${TBX_AGENT_REQUIRE_REAL_INFERENCE:-true}
REQUIRE_LLM=${TBX_AGENT_REQUIRE_LLM_INFERENCE:-true}
LLM_CONFIG=${TBX_AGENT_LLM_RUNTIME_CONFIG:-"$PROJECT_ROOT/configs/llm_runtime.yaml"}
API_ONLY=0
SKIP_PREFLIGHT=0
DRY_RUN=0
DEMO=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --api-only) API_ONLY=1 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --demo) DEMO=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            echo "Usage: scripts/run_local.sh [--api-only] [--skip-preflight] [--demo] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$DEMO" -eq 1 ]; then
    # Explicitly labelled control-flow demo. Apply after .env so a real-runtime
    # profile cannot accidentally leak into a mock demonstration.
    VISION_BACKEND=mock
    NARRATOR_BACKEND=none
    REQUIRE_VISION=false
    REQUIRE_LLM=false
    SKIP_PREFLIGHT=1
    export TBX_AGENT_VISION_BACKEND=$VISION_BACKEND
    export TBX_AGENT_NARRATOR_BACKEND=$NARRATOR_BACKEND
    export LLM_PROVIDER=none
    export TBX_AGENT_RETRIEVAL_CONFIG="$PROJECT_ROOT/configs/retrieval.yaml"
    export TBX_AGENT_REQUIRE_REAL_INFERENCE=$REQUIRE_VISION
    export TBX_AGENT_REQUIRE_LLM_INFERENCE=$REQUIRE_LLM
    export TBX_AGENT_OPENAI_ENABLED=false
    export TBX_AGENT_ANATOMY_BACKEND=none
    export TBX_AGENT_REQUIRE_ANATOMY_INFERENCE=false
    export TBX_AGENT_CONTOUR_REFINEMENT_BACKEND=none
fi

if truthy "$REQUIRE_VISION" && [ "$VISION_BACKEND" != rank03 ]; then
    echo "Real inference is required, but vision backend is '$VISION_BACKEND'." >&2
    exit 2
fi
if truthy "$REQUIRE_LLM" && [ "$NARRATOR_BACKEND" != llama_cpp ]; then
    echo "LLM inference is required, but narrator is '$NARRATOR_BACKEND'." >&2
    exit 2
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Project: $PROJECT_ROOT"
    echo "Runtime data: $TBX_AGENT_DATA_ROOT"
    echo "API: $PYTHON -m uvicorn tbx_agent.api.main:app --host $BIND_HOST --port $BIND_PORT"
    [ "$API_ONLY" -eq 1 ] || echo "UI:  $PYTHON -m streamlit run ui/streamlit_app.py --server.address=127.0.0.1 --server.port=$UI_PORT"
    echo "Required contract: vision=$REQUIRE_VISION/$VISION_BACKEND, llm=$REQUIRE_LLM/$NARRATOR_BACKEND"
    [ "$DEMO" -eq 0 ] || echo "Mode: DEMO / MOCK (not real model inference)"
    truthy "$REQUIRE_LLM" && echo "LLM supervisor: $PYTHON -m tbx_agent.llm serve --config $LLM_CONFIG"
    echo "Dry run complete; no process or runtime file was created."
    exit 0
fi

if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
    echo "Running fail-closed asset/configuration preflight..."
    if ! "$PYTHON" -m tbx_agent.preflight; then
        echo "Preflight failed. Install the downloaded inference bundle with scripts/install_vision_bundle.py --bundle <zip> --artifact-root <path> --runtime-config <path>, verify the configured LLM, and review .env." >&2
        exit 2
    fi
fi

LOG_ROOT="$TBX_AGENT_DATA_ROOT/logs"
mkdir -p "$LOG_ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LLM_LOG="$LOG_ROOT/llama-$STAMP.log"
API_LOG="$LOG_ROOT/api-$STAMP.log"
UI_LOG="$LOG_ROOT/ui-$STAMP.log"
LLM_PID=""
API_PID=""
UI_PID=""

cleanup() {
    [ -z "$UI_PID" ] || kill "$UI_PID" 2>/dev/null || true
    [ -z "$API_PID" ] || kill "$API_PID" 2>/dev/null || true
    [ -z "$LLM_PID" ] || kill "$LLM_PID" 2>/dev/null || true
    [ -z "$UI_PID" ] || wait "$UI_PID" 2>/dev/null || true
    [ -z "$API_PID" ] || wait "$API_PID" 2>/dev/null || true
    [ -z "$LLM_PID" ] || wait "$LLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

if truthy "$REQUIRE_LLM" && [ "$NARRATOR_BACKEND" = llama_cpp ]; then
    if [ "$SKIP_PREFLIGHT" -eq 1 ]; then
        "$PYTHON" -m tbx_agent.llm verify --config "$LLM_CONFIG" >/dev/null
    fi
    if ! "$PYTHON" -m tbx_agent.llm probe --skip-assets --timeout 3 --config "$LLM_CONFIG" >/dev/null 2>&1; then
        "$PYTHON" -m tbx_agent.llm serve --config "$LLM_CONFIG" >"$LLM_LOG" 2>&1 &
        LLM_PID=$!
        READY=0
        i=0
        while [ "$i" -lt 180 ]; do
            if ! kill -0 "$LLM_PID" 2>/dev/null; then
                echo "llama.cpp exited during startup. See $LLM_LOG" >&2
                exit 2
            fi
            if "$PYTHON" -m tbx_agent.llm probe --skip-assets --timeout 2 --config "$LLM_CONFIG" >/dev/null 2>&1; then
                READY=1
                break
            fi
            i=$((i + 1))
            sleep 1
        done
        [ "$READY" -eq 1 ] || { echo "llama.cpp did not become ready. See $LLM_LOG" >&2; exit 2; }
    fi
fi

"$PYTHON" -m uvicorn tbx_agent.api.main:app --host "$BIND_HOST" --port "$BIND_PORT" >"$API_LOG" 2>&1 &
API_PID=$!

API_READY=0
i=0
while [ "$i" -lt 90 ]; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "API exited during startup. See $API_LOG" >&2
        exit 2
    fi
    if "$PYTHON" - "$API_URL/readyz" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=2).read()
PY
    then
        API_READY=1
        break
    fi
    i=$((i + 1))
    sleep 1
done
[ "$API_READY" -eq 1 ] || { echo "API did not become ready within 90 seconds. See $API_LOG" >&2; exit 2; }

if [ "$API_ONLY" -eq 0 ]; then
    "$PYTHON" -m streamlit run ui/streamlit_app.py --server.address=127.0.0.1 --server.port="$UI_PORT" >"$UI_LOG" 2>&1 &
    UI_PID=$!
fi

echo "API: $API_URL (readiness: $API_URL/readyz)"
[ "$API_ONLY" -eq 1 ] || echo "UI:  http://127.0.0.1:$UI_PORT"
echo "Logs: $LOG_ROOT"
echo "Press Ctrl+C to stop both processes."

while kill -0 "$API_PID" 2>/dev/null; do
    if [ -n "$UI_PID" ] && ! kill -0 "$UI_PID" 2>/dev/null; then
        echo "UI exited unexpectedly. See $UI_LOG" >&2
        exit 2
    fi
    sleep 2
done
echo "API exited unexpectedly. See $API_LOG" >&2
exit 2
