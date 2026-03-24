#!/usr/bin/env bash
# scripts/run_2f.sh
# Phase 2f — Evol-Instruct complexity evolution (evolve_conversations.py)
#
# Applies three evolution operators to existing conversations:
#   deepen     — add emotional depth, more layered escalation
#   diversify  — rewrite with different vocabulary / register
#   complicate — introduce a confounding low-risk narrative alongside high-risk signal
#
# Each evolved conversation is verified via a Constitutional AI self-check before
# being accepted. Only hackathon endpoint models used — no cost.
#
# Should run after run_1a.sh and/or run_1b.sh.
# Can run in parallel with run_2e.sh once Phase 1 CSVs exist.
# Can also run in parallel with run_2abcd.sh (uses different source data).
#
# Usage:
#   bash scripts/run_2f.sh                # full run  (120 evolved conversations)
#   bash scripts/run_2f.sh --test         # test run  (3 evolved conversations, verbose)
#   bash scripts/run_2f.sh --append       # resume    (reduces target by already-written rows)
#   bash scripts/run_2f.sh --test --append
#
# Env vars required:
#   BUZZ_MISTRAL_LARGE_AUTH_TOKEN  — Mistral endpoint token (from .env)
#   BUZZ_COHERE_AUTH_TOKEN         — Cohere/Command endpoint token (from .env)
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


# ── Check input CSVs ──────────────────────────────────────────────────────────
# evolve_conversations.py uses --inputs f1 f2 ... (nargs="+"), so we collect
# all available paths and pass them after a single --inputs flag.
INPUT_FILES=()

for f in datasets/generated_scratch.csv datasets/generated_camel.csv; do
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

# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --inputs "${INPUT_FILES[@]}"
  --output  datasets/evolved.csv
  --seed    42
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 3 total evolved conversations, verbose ON    ║"
  echo "║  (1 per operator: deepen / diversify / complicate)       ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--target 3 --verbose)
else
  CMD_ARGS+=(--target 120)
fi

if [[ $APPEND_MODE -eq 1 ]]; then
  echo "INFO: Append/resume mode enabled."
  CMD_ARGS+=(--append)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2f — Evol-Instruct complexity evolution"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  Operators : deepen, diversify, complicate"
echo "  Self-check: Constitutional AI (Mistral Large 3)"
echo "  Models    : Mistral Large (Mistral AI API) + Llama 3.3 70B (Together AI)"

mkdir -p datasets

python project/scripts/evolve_conversations.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 2f done — $(date)"
echo "  Output: datasets/evolved.csv"
