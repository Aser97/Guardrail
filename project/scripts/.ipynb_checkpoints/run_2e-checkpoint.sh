#!/usr/bin/env bash
# scripts/run_2e.sh
# Phase 2e — Linguistic realism degradation (degrade_register.py)
#
# Rule-based, no LLM calls, runs in seconds. Applies four stochastic
# transformations to user turns only:
#   - Sentence fragmentation
#   - Abbreviation injection
#   - Affect flattening (punctuation removal)
#   - Typo injection
#
# Should run after run_1a.sh and run_1b.sh have produced output CSVs.
# Can run in parallel with run_2f.sh once Phase 1 CSVs exist.
#
# Usage:
#   bash scripts/run_2e.sh          # standard run (rate=30%)
#   bash scripts/run_2e.sh --test   # same as standard (rule-based, no LLM — always fast)
#
# No env vars required (no LLM calls).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Source .env if present ────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  set -o allexport; source .env; set +o allexport
  echo "INFO: Loaded .env"
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
TEST_MODE=0
for arg in "$@"; do
  case "$arg" in
    --test)   TEST_MODE=1 ;;
    --append) echo "INFO: --append not applicable for Phase 2e (rule-based, idempotent run). Ignored." ;;
    *)
      echo "ERROR: Unknown argument: $arg"
      echo "       Usage: $0 [--test]"
      exit 1
      ;;
  esac
done

# ── Check input CSVs ─────────────────────────────────────────────────────────
# degrade_register.py uses --inputs f1 f2 ... (nargs="+"), so we collect
# all available paths into a single array and pass them after one --inputs flag.
INPUT_FILES=()

for f in datasets/generated_scratch.csv \
         datasets/generated_camel.csv \
         datasets/generated_camel_hard.csv \
         datasets/generated_pair.csv \
         datasets/generated_pair_hard.csv; do
  if [[ -f "$f" ]]; then
    INPUT_FILES+=("$f")
  else
    echo "WARN: $f not found — will be skipped."
  fi
done

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No input CSVs found."
  echo "       Run run_1a.sh and/or run_1b.sh first to generate Phase 1 data."
  exit 1
fi

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: rule-based script — runs at full speed       ║"
  echo "║  Output will still be inspectable in datasets/degraded.csv ║"
  echo "╚══════════════════════════════════════════════════════════╝"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2e — Linguistic realism degradation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  Transformations: fragmentation, abbreviations,"
echo "                   affect-flattening, typo injection"
echo "  Degradation rate: 30%"
echo "  Inputs: ${INPUT_FILES[*]}"

mkdir -p datasets

python project/scripts/degrade_register.py \
  --inputs "${INPUT_FILES[@]}" \
  --output datasets/degraded.csv \
  --rate   0.30 \
  --seed   42

echo ""
echo "✓ Phase 2e done — $(date)"
echo "  Output: datasets/degraded.csv"
