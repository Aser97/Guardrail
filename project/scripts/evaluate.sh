#!/usr/bin/env bash
# Eval script: evaluate predictions for a team and write eval metrics.
# Usage: scripts/evaluate.sh <path_to_predictions.csv> <path_to_eval_metrics.csv>
#
# Reads the team's predictions CSV and writes evaluation metrics (precision,
# recall, F1, latency) to eval_metrics.csv.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/evaluate.sh <path_to_predictions.csv> <path_to_eval_metrics.csv>" >&2
  exit 1
fi

PREDICTIONS_CSV="$1"
EVAL_METRICS_CSV="$2"

# Invocation dir: where the user ran the script from (e.g. repo root)
INVOKED_DIR="$(pwd)"
# Project root = parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve paths relative to invocation dir so "results/predictions.csv" from repo root works after we cd to project/
[[ "$PREDICTIONS_CSV" != /* ]] && PREDICTIONS_CSV="$INVOKED_DIR/$PREDICTIONS_CSV"
[[ "$EVAL_METRICS_CSV" != /* ]] && EVAL_METRICS_CSV="$INVOKED_DIR/$EVAL_METRICS_CSV"

if [[ ! -f "$PREDICTIONS_CSV" ]]; then
  echo "Predictions CSV not found: $PREDICTIONS_CSV" >&2
  exit 1
fi

# get_guardrail_metrics writes metrics.csv to the same dir as --output (JSON)
OUTPUT_DIR="$(dirname "$EVAL_METRICS_CSV")"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
PYTHONPATH=. python -m src.guardrails.get_guardrail_metrics \
  --predictions "$PREDICTIONS_CSV" \
  --output "$OUTPUT_DIR/eval_metrics.json"

# Move metrics.csv to the requested path (skip if already there)
if [[ -f "$OUTPUT_DIR/metrics.csv" ]]; then
  if [[ "$OUTPUT_DIR/metrics.csv" != "$EVAL_METRICS_CSV" ]]; then
    mv "$OUTPUT_DIR/metrics.csv" "$EVAL_METRICS_CSV"
  fi
  echo "Eval metrics written to $EVAL_METRICS_CSV"
else
  echo "Metrics CSV was not produced." >&2
  exit 1
fi
