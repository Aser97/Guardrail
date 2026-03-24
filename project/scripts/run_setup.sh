#!/usr/bin/env bash
# project/scripts/run_setup.sh
# One-time setup — build persona bank and preprocess ESConv dataset.
#
# Run this ONCE before any of the data-generation scripts (run_1a, run_1b, etc.)
# Re-running is safe: both steps skip work if the output file already exists.
#
# Usage:
#   bash project/scripts/run_setup.sh              # full run (2 000 personas, 400 ESConv rows)
#   bash project/scripts/run_setup.sh --test       # quick smoke test (50 personas, 20 ESConv rows)
#   bash project/scripts/run_setup.sh --skip-esconv  # skip ESConv (if you don't have the file)
#
# Env vars: none required for this script.

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
SKIP_ESCONV=0
for arg in "$@"; do
  case $arg in
    --test)        TEST_MODE=1 ;;
    --skip-esconv) SKIP_ESCONV=1 ;;
  esac
done

if [[ $TEST_MODE -eq 1 ]]; then
  PERSONA_TOTAL=50
  ESCONV_MAX=20
  echo "INFO: TEST MODE — persona_total=$PERSONA_TOTAL  esconv_max=$ESCONV_MAX"
else
  PERSONA_TOTAL=2000
  ESCONV_MAX=400
fi

# ── Step 1: Build persona bank ────────────────────────────────────────────────
PERSONA_OUT="datasets/persona_bank.json"

if [[ -f "$PERSONA_OUT" ]]; then
  echo "INFO: $PERSONA_OUT already exists — skipping persona bank generation."
else
  echo "INFO: Building persona bank ($PERSONA_TOTAL personas) → $PERSONA_OUT"
  python project/scripts/build_persona_bank.py \
    --output "$PERSONA_OUT" \
    --total  "$PERSONA_TOTAL"
  echo "OK:   Persona bank written to $PERSONA_OUT"
fi

# ── Step 2: Preprocess ESConv ─────────────────────────────────────────────────
ESCONV_IN="datasets/esconv.json"
ESCONV_OUT="datasets/esconv_preprocessed.csv"

if [[ $SKIP_ESCONV -eq 1 ]]; then
  echo "INFO: --skip-esconv set — skipping ESConv preprocessing."
elif [[ ! -f "$ESCONV_IN" ]]; then
  echo "WARN: $ESCONV_IN not found — skipping ESConv preprocessing."
  echo "      Download it from https://github.com/thu-coai/Emotional-Support-Conversation"
  echo "      and place it at datasets/esconv.json, then re-run this script."
elif [[ -f "$ESCONV_OUT" ]]; then
  echo "INFO: $ESCONV_OUT already exists — skipping ESConv preprocessing."
else
  echo "INFO: Preprocessing ESConv (max $ESCONV_MAX rows) → $ESCONV_OUT"
  python project/scripts/preprocess_esconv.py \
    --input    "$ESCONV_IN" \
    --output   "$ESCONV_OUT" \
    --max_rows "$ESCONV_MAX"
  echo "OK:   ESConv preprocessed → $ESCONV_OUT"
fi

echo ""
echo "✓ Setup complete. You can now run the data-generation scripts:"
echo "    bash project/scripts/run_1a.sh"
echo "    bash project/scripts/run_1b.sh"
echo "    bash project/scripts/run_2abcd.sh"
