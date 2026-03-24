#!/usr/bin/env bash
# scripts/run_1a.sh
# Phase 1a — From-scratch generation (generate_scratch.py)
#
# Can run independently and in parallel with run_1b.sh and run_2abcd.sh.
# Supports incremental writes + resume across Paperspace Gradient sessions.
#
# Usage:
#   bash project/scripts/run_1a.sh                # full run  (540 high-risk + 200 low-risk)
#   bash project/scripts/run_1a.sh --test         # test run  (2 per signal + 3 low-risk, verbose)
#   bash project/scripts/run_1a.sh --append       # resume    (skips already-written rows)
#   bash project/scripts/run_1a.sh --test --append
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
if [[ -z "${BUZZ_MISTRAL_LARGE_AUTH_TOKEN:-}" ]] && [[ -z "${BUZZ_COHERE_AUTH_TOKEN:-}" ]]; then
  echo "ERROR: No hackathon API tokens found."
  echo "       Ensure your .env file contains BUZZ_MISTRAL_LARGE_AUTH_TOKEN and BUZZ_COHERE_AUTH_TOKEN."
  exit 1
fi


# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --output   datasets/generated_scratch.csv
  --seed     42
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 2 per signal, 3 low-risk, verbose ON         ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--per_signal 2 --low_risk 3 --verbose)
else
  CMD_ARGS+=(--per_signal 60 --low_risk 200)
fi

if [[ $APPEND_MODE -eq 1 ]]; then
  echo "INFO: Append/resume mode enabled."
  CMD_ARGS+=(--append)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 1a — From-scratch generation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"

mkdir -p datasets

python project/scripts/generate_scratch.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 1a done — $(date)"
echo "  Output: datasets/generated_scratch.csv"
