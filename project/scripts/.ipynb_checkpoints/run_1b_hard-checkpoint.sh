#!/usr/bin/env bash
# scripts/run_1b_hard.sh
# Phase 1b-hard — CAMEL dual-agent generation with enforced hardness constraints.
#
# Two hardness tracks:
#   subtle     — signal present but always expressed indirectly (subtlety ≥ 7 gate)
#   escalating — arc starts low-risk, builds to signal peak (escalation_arc ≥ 7 gate)
#
# Both tracks use a chain-of-thought Claude Sonnet judge that reasons step-by-step
# before scoring. Acceptance requires ALL gated criteria to be met.
#
# USER role    : Mistral Large     (Mistral AI API — voices person in distress)
# SUPPORT role : Llama 3.3 70B    (Together AI    — voices AI support chat)
# JUDGE        : Claude Sonnet     (Anthropic API  — CoT quality gate)
# ANNOTATION   : Mistral Large     (Mistral AI API — labels each turn)
#
# Usage:
#   bash scripts/run_1b_hard.sh                # full run  (300 conversations, 50/50 split)
#   bash scripts/run_1b_hard.sh --test         # test run  (4 conversations, verbose)
#   bash scripts/run_1b_hard.sh --append       # resume
#   bash scripts/run_1b_hard.sh --test --append
#
# Env vars required:
#   MISTRAL_API_KEY   — Mistral AI API key (console.mistral.ai)
#   ANTHROPIC_API_KEY — Anthropic API key  (console.anthropic.com)
#   TOGETHER_API_KEY  — Together AI key    (api.together.xyz)

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
if [[ -z "${TOGETHER_API_KEY:-}" ]]; then
  echo "ERROR: TOGETHER_API_KEY not found. Add it to your .env file."
  exit 1
fi

# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --output     datasets/generated_camel_hard.csv
  --subtle_frac 0.5
  --seed       42
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 4 conversations (2 subtle + 2 escalating), verbose  ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--target 4 --verbose)
else
  CMD_ARGS+=(--target 300)
fi

if [[ $APPEND_MODE -eq 1 ]]; then
  echo "INFO: Append/resume mode enabled."
  CMD_ARGS+=(--append)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 1b-hard — CAMEL Hard Generation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  USER role    : Mistral Large     (Mistral AI API)"
echo "  SUPPORT role : Llama 3.3 70B    (Together AI)"
echo "  JUDGE        : Claude Sonnet     (Anthropic API — CoT)"
echo "  Tracks       : subtle (50%) + escalating (50%)"
echo "  Judge gates  : signal≥7, realism≥7, subtlety≥7 | escalation_arc≥7"

mkdir -p datasets

python project/scripts/generate_camel_hard.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 1b-hard done — $(date)"
echo "  Output: datasets/generated_camel_hard.csv"
