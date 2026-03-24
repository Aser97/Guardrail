#!/usr/bin/env bash
# scripts/run_2g.sh
# Phase 2g — PAIR adversarial generation (generate_pair.py)
#
# Two modes per row:
#   hard_positive        — signal present but expressed INDIRECTLY (subtle)
#   adversarial_negative — mimics high-risk surface language but is genuinely low-risk
#
# Generator : Mistral Large (Mistral AI API — private key)
# Judge     : Claude 3.7 Sonnet (Anthropic API — private key)
#
# The PAIR loop retries up to --max_iter times, feeding judge feedback back
# to the generator on each rejection.
#
# Should run after Phase 1 data exists (can run in parallel with 2a-2f).
#
# Usage:
#   bash project/scripts/run_2g.sh                # full run  (200 conversations)
#   bash project/scripts/run_2g.sh --test         # test run  (4 conversations, verbose)
#   bash project/scripts/run_2g.sh --append       # resume    (reduces targets by already-written rows)
#   bash project/scripts/run_2g.sh --test --append
#
# Env vars required:
#   MISTRAL_API_KEY   — Mistral AI API key (console.mistral.ai)
#   ANTHROPIC_API_KEY — Anthropic API key  (console.anthropic.com)

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
APPEND_MODE=0
for arg in "$@"; do
  case "$arg" in
    --test)   TEST_MODE=1   ;;
    --append) APPEND_MODE=1 ;;
    *)
      echo "ERROR: Unknown argument: $arg"
      echo "       Usage: $0 [--test] [--append]"
      exit 1
      ;;
  esac
done

# ── Check required env vars ───────────────────────────────────────────────────
if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
  echo "ERROR: MISTRAL_API_KEY not found."
  echo "       Get a key at console.mistral.ai and add MISTRAL_API_KEY=... to your .env file."
  exit 1
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY not found."
  echo "       Get a key at console.anthropic.com and add ANTHROPIC_API_KEY=... to your .env file."
  exit 1
fi

# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --output  datasets/generated_pair.csv
  --max_iter 5
  --seed    42
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 4 total conversations, verbose ON            ║"
  echo "║  (2 hard positives + 2 adversarial negatives)            ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--target 4 --verbose)
else
  CMD_ARGS+=(--target 200)
fi

if [[ $APPEND_MODE -eq 1 ]]; then
  echo "INFO: Append/resume mode enabled."
  CMD_ARGS+=(--append)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2g — PAIR adversarial generation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  Generator : Mistral Large (Mistral AI API)"
echo "  Judge     : Claude 3.7 Sonnet (Anthropic API)"
echo "  Modes     : hard_positive (70%) + adversarial_negative (30%)"

mkdir -p datasets

python project/scripts/generate_pair.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 2g done — $(date)"
echo "  Output: datasets/generated_pair.csv"
echo "  Attempt log: datasets/generated_pair.attempts.json"
