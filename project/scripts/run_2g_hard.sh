#!/usr/bin/env bash
# scripts/run_2g_hard.sh
# Phase 2g-hard — PAIR adversarial generation with enforced hardness constraints.
#
# Three tracks:
#   subtle              — signal present but expressed indirectly (subtlety ≥ 7)
#   escalating          — arc starts low-risk, builds to signal peak (escalation_arc ≥ 7)
#   adversarial_negative — looks high-risk on surface, definitively low-risk on reading
#                          (mimicry ≥ 7 AND absence_clarity ≥ 8)
#
# All tracks use chain-of-thought Claude Sonnet judge with explicit step-by-step
# reasoning before scoring. Stricter than run_2g.sh.
#
# Generator : Mistral Large  (Mistral AI API — private key)
# Judge     : Claude Sonnet  (Anthropic API — private key, CoT)
#
# Usage:
#   bash scripts/run_2g_hard.sh                # full run (200 conversations)
#   bash scripts/run_2g_hard.sh --test         # test run (6 conversations, verbose)
#   bash scripts/run_2g_hard.sh --append       # resume
#   bash scripts/run_2g_hard.sh --test --append
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
  echo "ERROR: MISTRAL_API_KEY not found. Add it to your .env file."
  exit 1
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY not found. Add it to your .env file."
  exit 1
fi

# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --output          datasets/generated_pair_hard.csv
  --max_iter        5
  --subtle_frac     0.40
  --escalating_frac 0.40
  --seed            1337
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 6 total (2 subtle + 2 escalating + 2 neg), verbose ON  ║"
  echo "╚══════════════════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--target 6 --verbose)
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
echo " Phase 2g-hard — PAIR Hard Adversarial Generation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  Generator : Mistral Large  (Mistral AI API)"
echo "  Judge     : Claude Sonnet  (Anthropic API — CoT)"
echo "  Tracks    : subtle (40%) + escalating (40%) + adversarial_negative (20%)"
echo "  Pos gates : signal≥7, realism≥7, subtlety≥7 | escalation_arc≥7"
echo "  Neg gates : mimicry≥7, absence_clarity≥8, realism≥7"

mkdir -p datasets

python project/scripts/generate_pair_hard.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 2g-hard done — $(date)"
echo "  Output:      datasets/generated_pair_hard.csv"
echo "  Attempt log: datasets/generated_pair_hard.attempts.json"
