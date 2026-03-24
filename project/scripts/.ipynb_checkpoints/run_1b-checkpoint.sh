#!/usr/bin/env bash
# scripts/run_1b.sh
# Phase 1b — CAMEL dual-agent generation (generate_camel.py)
#
# Can run independently and in parallel with run_1a.sh and run_2abcd.sh.
# Supports incremental writes + resume across Paperspace Gradient sessions.
#
# Usage:
#   bash project/scripts/run_1b.sh                # full run  (300 conversations)
#   bash project/scripts/run_1b.sh --test         # test run  (3 conversations, verbose)
#   bash project/scripts/run_1b.sh --append       # resume    (skips already-written rows)
#   bash project/scripts/run_1b.sh --test --append
#
# Env vars required:
#   MISTRAL_API_KEY   — Mistral AI API key (console.mistral.ai) — user/annotation role
#   TOGETHER_API_KEY  — Together AI API key (together.ai)       — assistant role
# Env vars optional:


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
if [[ -z "${TOGETHER_API_KEY:-}" ]]; then
  echo "ERROR: TOGETHER_API_KEY not found."
  echo "       Get a key at together.ai and add TOGETHER_API_KEY=... to your .env file."
  exit 1
fi


# ── Persona bank check ────────────────────────────────────────────────────────
if [[ ! -f "datasets/persona_bank.json" ]]; then
  echo "INFO: Persona bank not found — building it now."
  mkdir -p datasets
  python project/scripts/build_persona_bank.py --output datasets/persona_bank.json --total 2000
  echo "✓ Persona bank built."
fi

# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --output  datasets/generated_camel.csv
  --seed    42
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 3 conversations, verbose ON                  ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--target 3 --verbose)
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
echo " Phase 1b — CAMEL dual-agent generation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  USER role    : Mistral Large     (Mistral AI API — voices person in distress)"
echo "  SUPPORT role : Llama 3.3 70B    (Together AI    — voices AI support chat)"
echo "  ANNOTATION   : Mistral Large     (Mistral AI API — labels each turn)"

mkdir -p datasets

python project/scripts/generate_camel.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 1b done — $(date)"
echo "  Output: datasets/generated_camel.csv"
