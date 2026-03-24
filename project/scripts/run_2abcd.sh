#!/usr/bin/env bash
# scripts/run_2abcd.sh
# Phase 2a-2d — Seed augmentation (augment_seed.py)
#
#   2a — Language rewrite (translate to French/Spanish/Portuguese)
#   2b — Persona swap (KHP + ESConv-based)
#   2c — Signal injection (low-risk → borderline/high-risk)
#   2d — Signal softening (high-risk → borderline)
#
# Can run independently and in parallel with run_1a.sh and run_1b.sh.
# Requires: datasets/seed_validation_set.csv (always present in repo)
# Optional: datasets/esconv_preprocessed.csv (if ESConv was downloaded)
# Optional: datasets/persona_bank.json       (built automatically if missing)
#
# Usage:
#   bash project/scripts/run_2abcd.sh                # full run  (~80 rows per augmentation type)
#   bash project/scripts/run_2abcd.sh --test         # test run  (2 per type, verbose)
#   bash project/scripts/run_2abcd.sh --append       # resume    (proportional skip of existing rows)
#   bash project/scripts/run_2abcd.sh --test --append
#
# Env vars required:
#   MISTRAL_API_KEY   — Mistral AI API key (console.mistral.ai)
#   ANTHROPIC_API_KEY — Anthropic API key  (console.anthropic.com)
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
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY not found."
  echo "       Get a key at console.anthropic.com and add ANTHROPIC_API_KEY=... to your .env file."
  exit 1
fi


# ── Check seed CSV ────────────────────────────────────────────────────────────
if [[ ! -f "datasets/seed_validation_set.csv" ]]; then
  echo "ERROR: datasets/seed_validation_set.csv not found."
  echo "       This file must be present in the repo (it is the human-annotated seed set)."
  exit 1
fi

# ── Persona bank check ────────────────────────────────────────────────────────
if [[ ! -f "datasets/persona_bank.json" ]]; then
  echo "INFO: Persona bank not found — building it now."
  mkdir -p datasets
  python project/scripts/build_persona_bank.py --output datasets/persona_bank.json --total 2000
  echo "✓ Persona bank built."
fi

# ── ESConv check (optional) ───────────────────────────────────────────────────
ESCONV_ARG=""
if [[ -f "datasets/esconv_preprocessed.csv" ]]; then
  ESCONV_ARG="--esconv_csv datasets/esconv_preprocessed.csv"
  echo "INFO: ESConv preprocessed data found — will be used for persona-swap augmentation."
elif [[ -f "datasets/esconv.json" ]]; then
  echo "INFO: Raw esconv.json found — preprocessing now."
  python project/scripts/preprocess_esconv.py \
    --input  datasets/esconv.json \
    --output datasets/esconv_preprocessed.csv \
    --max_rows 400
  ESCONV_ARG="--esconv_csv datasets/esconv_preprocessed.csv"
  echo "✓ ESConv preprocessed."
else
  echo "INFO: No ESConv data found — Phase 2b/2c ESConv-swap steps will be skipped."
  echo "      Download from: https://github.com/thu-coai/Emotional-Support-Conversation"
fi

# ── Build arguments ───────────────────────────────────────────────────────────
CMD_ARGS=(
  --seed_csv     datasets/seed_validation_set.csv
  --persona_bank datasets/persona_bank.json
  --output       datasets/augmented.csv
  --seed         42
)

# Append ESConv arg only if we have the file
if [[ -n "$ESCONV_ARG" ]]; then
  # shellcheck disable=SC2206
  CMD_ARGS+=($ESCONV_ARG)
fi

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 2 per augmentation type, verbose ON          ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  CMD_ARGS+=(--per_type 2 --esconv_swap_target 2 --verbose)
else
  CMD_ARGS+=(--per_type 80 --esconv_swap_target 300)
fi

if [[ $APPEND_MODE -eq 1 ]]; then
  echo "INFO: Append/resume mode enabled."
  CMD_ARGS+=(--append)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2a-2d — Seed augmentation"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"
echo "  2a — Language rewrite  (French / Spanish / Portuguese)"
echo "  2b — Persona swap      (KHP + ESConv)"
echo "  2c — Signal injection  (low-risk → high-risk)"
echo "  2d — Signal softening  (high-risk → borderline)"
echo "  Models: Mistral Large (Mistral AI API) + Claude Haiku (Anthropic API)"

mkdir -p datasets

python project/scripts/augment_seed.py "${CMD_ARGS[@]}"

echo ""
echo "✓ Phase 2a-2d done — $(date)"
echo "  Output: datasets/augmented.csv"
