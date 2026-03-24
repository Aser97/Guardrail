#!/usr/bin/env bash
# Predict script: fixed name and path for hackathon submissions.
# Usage: scripts/predict.sh <path_to_input_file.csv> <path_to_predictions_output_file.csv>
#
# Reads the input CSV and writes predictions to the output CSV using this team's
# guardrail (src.guardrails.get_predictions).

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/predict.sh <path_to_input_file.csv> <path_to_predictions_output_file.csv>" >&2
  exit 1
fi

INPUT_CSV="$1"
OUTPUT_CSV="$2"

# Invocation dir: where the user ran the script from (e.g. repo root)
INVOKED_DIR="$(pwd)"
# Project root = parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve paths relative to invocation dir so "datasets/foo.csv" from repo root works after we cd to project/
[[ "$INPUT_CSV" != /* ]] && INPUT_CSV="$INVOKED_DIR/$INPUT_CSV"
[[ "$OUTPUT_CSV" != /* ]] && OUTPUT_CSV="$INVOKED_DIR/$OUTPUT_CSV"

# Fixed user-owned submission module path for deterministic local/evaluator behavior.
SUBMISSION_MODULE="src/submission/submission.py"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

# get_predictions writes to <output_dir>/predictions.csv; we then move to requested path
OUTPUT_DIR="$(dirname "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
PYTHONPATH=. python -m src.guardrails.get_predictions \
  --submission "$SUBMISSION_MODULE" \
  --data "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR"

# Write to the exact path requested (script writes predictions.csv into output-dir)
if [[ -f "$OUTPUT_DIR/predictions.csv" ]]; then
  if [[ "$OUTPUT_DIR/predictions.csv" != "$OUTPUT_CSV" ]]; then
    mv "$OUTPUT_DIR/predictions.csv" "$OUTPUT_CSV"
  fi
  echo "Predictions written to $OUTPUT_CSV"
else
  echo "Predictions file was not produced." >&2
  exit 1
fi
