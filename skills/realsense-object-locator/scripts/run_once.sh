#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="config.yaml"
JSON_OUT=""
RESULT_JSON=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: run_once.sh [--config config.yaml] [--result-json 'runs/results/{run_id}.json'] [--json-out path] [--] [extra object-locator args...]

Runs the object-locator command from the repository root inferred from this skill.
By default, config.yaml saves non-overwriting results through output.result_json.

Examples:
  skills/realsense-object-locator/scripts/run_once.sh
  skills/realsense-object-locator/scripts/run_once.sh --result-json 'runs/results/{run_id}.json'
  skills/realsense-object-locator/scripts/run_once.sh -- --detector grounded_sam --json
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --json-out)
      JSON_OUT="$2"
      shift 2
      ;;
    --result-json)
      RESULT_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$REPO_ROOT"
mkdir -p runs

if [[ -n "$RESULT_JSON" ]]; then
  EXTRA_ARGS+=(--result-json "$RESULT_JSON")
fi

if [[ -n "$JSON_OUT" ]]; then
  mkdir -p "$(dirname "$JSON_OUT")"
  object-locator --config "$CONFIG" --json "${EXTRA_ARGS[@]}" > "$JSON_OUT"
  printf 'Wrote JSON result to %s\n' "$JSON_OUT"
else
  object-locator --config "$CONFIG" "${EXTRA_ARGS[@]}"
fi
