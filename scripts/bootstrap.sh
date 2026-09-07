#!/usr/bin/env sh
set -eu
umask 077

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
VENV=${VENV:-"$PROJECT_ROOT/.venv"}
EXTRAS=${TBX_INSTALL_EXTRAS:-ui,dicom}
CACHE_DIR=${TBX_ARTIFACT_ROOT:-}
VISION_BUNDLE=""
RUNTIME_CONFIG=""
SKIP_MODELS=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/bootstrap.sh [options]
  --python PATH          Python 3.11-3.13 executable
  --venv PATH            virtual environment (default: .venv)
  --extras LIST          install extras (default: ui,dicom; no model downloads)
  --cache-dir PATH       external artifact cache
  --vision-bundle ZIP    install a manually downloaded inference release bundle
  --runtime-config PATH required external runtime JSON output with --vision-bundle
  --skip-models          compatibility flag; models are not downloaded by default
  --dry-run             print actions without changing files
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python|--venv|--extras|--cache-dir|--vision-bundle|--runtime-config)
            option=$1
            [ "$#" -ge 2 ] || { echo "$option requires a value." >&2; exit 2; }
            case "$option" in
                --python) PYTHON=$2 ;;
                --venv) VENV=$2 ;;
                --extras) EXTRAS=$2 ;;
                --cache-dir) CACHE_DIR=$2 ;;
                --vision-bundle) VISION_BUNDLE=$2 ;;
                --runtime-config) RUNTIME_CONFIG=$2 ;;
            esac
            shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [ -z "$CACHE_DIR" ]; then
    if [ -n "${XDG_CACHE_HOME:-}" ]; then
        CACHE_DIR="$XDG_CACHE_HOME/tbx-agent/artifacts"
    elif [ -n "${HOME:-}" ]; then
        CACHE_DIR="$HOME/.cache/tbx-agent/artifacts"
    else
        echo 'Set --cache-dir or TBX_ARTIFACT_ROOT when HOME and XDG_CACHE_HOME are unset.' >&2
        exit 2
    fi
fi
case "$CACHE_DIR" in /*) ;; *) CACHE_DIR="$PWD/$CACHE_DIR" ;; esac
if [ -n "$VISION_BUNDLE" ]; then
    [ "$SKIP_MODELS" -eq 0 ] || { echo '--vision-bundle conflicts with --skip-models.' >&2; exit 2; }
    [ -n "$RUNTIME_CONFIG" ] || { echo '--vision-bundle requires --runtime-config.' >&2; exit 2; }
    case ",$EXTRAS," in *,vision,*) ;; *) EXTRAS="$EXTRAS,vision" ;; esac
    case "$VISION_BUNDLE" in /*) ;; *) VISION_BUNDLE="$PWD/$VISION_BUNDLE" ;; esac
    case "$RUNTIME_CONFIG" in /*) ;; *) RUNTIME_CONFIG="$PWD/$RUNTIME_CONFIG" ;; esac
fi
case "$VENV" in /*) ;; *) VENV="$PROJECT_ROOT/$VENV" ;; esac
VENV_PYTHON="$VENV/bin/python"

step() {
    printf '> %s\n' "$*"
    if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

if [ "$DRY_RUN" -eq 1 ] || [ ! -x "$VENV_PYTHON" ]; then
    step "$PYTHON" -m venv "$VENV"
fi
step "$VENV_PYTHON" -m pip install --upgrade pip
step "$VENV_PYTHON" -m pip install -e "$PROJECT_ROOT[$EXTRAS]"
if [ -n "$VISION_BUNDLE" ]; then
    step "$VENV_PYTHON" "$PROJECT_ROOT/scripts/install_vision_bundle.py" \
        --bundle "$VISION_BUNDLE" --artifact-root "$CACHE_DIR" --runtime-config "$RUNTIME_CONFIG"
fi
if [ "$DRY_RUN" -eq 1 ]; then
    echo 'Dry run complete; no files changed or models downloaded.'
    exit 0
fi
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    printf '\nTBX_ARTIFACT_ROOT=%s\n' "$CACHE_DIR" >> "$PROJECT_ROOT/.env"
    if [ -n "$VISION_BUNDLE" ]; then
        printf 'TBX_AGENT_RANK03_RUNTIME_CONFIG=%s\n' "$RUNTIME_CONFIG" >> "$PROJECT_ROOT/.env"
    fi
    echo 'Created .env; review it before real inference.'
else
    echo 'Existing .env was preserved.'
    if [ -n "$VISION_BUNDLE" ]; then
        echo "Set TBX_AGENT_RANK03_RUNTIME_CONFIG=$RUNTIME_CONFIG and TBX_ARTIFACT_ROOT=$CACHE_DIR in .env."
    fi
fi
echo 'Ready for a model-free demonstration: ./scripts/run_local.sh --demo'
echo 'For real vision, download the published inference bundle and use --vision-bundle <zip> --runtime-config <external-json>.'
echo 'Install optional D-FINE source and PSPNet explicitly with bootstrap_dfine.py and bootstrap_models.py download anatomy.'
