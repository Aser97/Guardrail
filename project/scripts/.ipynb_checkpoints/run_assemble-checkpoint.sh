#!/usr/bin/env bash
# scripts/run_assemble.sh
# Phase 2g + Assembly + Phase 3 gap-fill
#
#   Step 1: Phase 2g — PAIR adversarial loop (generate_pair.py)
#           Attacker: Mistral Large 3 (hackathon endpoint, free)
#           Judge:    Claude 3.7 Sonnet (Anthropic API, est. $20-30)
#
#   Step 2: Assemble all generated CSVs into master.csv + train.csv
#           (build_master_csv.py)
#
#   Step 3: Gap analysis — check per-signal coverage
#           (gap_analysis.py)
#
#   Step 4: Phase 3 gap-fill — targeted from-scratch generation for
#           any signal below MIN_PER_SIGNAL threshold (generate_scratch.py)
#
#   Step 5: Re-assemble if gap-fill was needed
#
# IMPORTANT: Run this AFTER all generation scripts (run_1a, run_1b,
#            run_2abcd, run_2e, run_2f) have completed. Those can be
#            run in any order and in parallel with each other.
#
# Usage:
#   bash scripts/run_assemble.sh                  # full run
#   bash scripts/run_assemble.sh --test           # test run (3 PAIR pairs, verbose)
#   bash scripts/run_assemble.sh --append         # resume PAIR generation
#   bash scripts/run_assemble.sh --skip-pair      # skip 2g (if already done or too costly)
#   bash scripts/run_assemble.sh --test --append
#
# Env vars required:
#   HACKATHON_API_TOKEN   — hackathon OpenAI-compat endpoint token
#   ANTHROPIC_API_KEY     — Anthropic API key (for generate_pair.py judge)
# Env vars optional:
#   HACKATHON_API_BASE    — defaults to Cohere compat endpoint if unset

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
SKIP_PAIR=0
for arg in "$@"; do
  case "$arg" in
    --test)      TEST_MODE=1   ;;
    --append)    APPEND_MODE=1 ;;
    --skip-pair) SKIP_PAIR=1   ;;
    *)
      echo "ERROR: Unknown argument: $arg"
      echo "       Usage: $0 [--test] [--append] [--skip-pair]"
      exit 1
      ;;
  esac
done

# ── Check env vars ────────────────────────────────────────────────────────────
if [[ -z "${BUZZ_MISTRAL_LARGE_AUTH_TOKEN:-}" ]] && [[ -z "${BUZZ_COHERE_AUTH_TOKEN:-}" ]]; then
  echo "ERROR: No hackathon API tokens found."
  echo "       Ensure your .env file contains BUZZ_MISTRAL_LARGE_AUTH_TOKEN and BUZZ_COHERE_AUTH_TOKEN."
  exit 1
fi

if [[ $SKIP_PAIR -eq 0 ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  echo "       export ANTHROPIC_API_KEY=<your-key>"
  echo "       (Required by Claude 3.7 Sonnet judge in generate_pair.py)"
  echo "       To skip Phase 2g, pass --skip-pair"
  exit 1
fi

echo ""
echo "============================================================"
echo " Assembly pipeline: Phase 2g + Assemble + Phase 3 gap-fill"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"

mkdir -p datasets

# ── Step 1: Phase 2g — PAIR adversarial loop ─────────────────────────────────
if [[ $SKIP_PAIR -eq 1 ]]; then
  echo ""
  echo "── Step 1: Phase 2g — PAIR (SKIPPED via --skip-pair) ──────"
else
  echo ""
  echo "── Step 1: Phase 2g — PAIR adversarial generation ─────────"
  echo "   Generator : Mistral Large 3 (hackathon endpoint)"
  echo "   Judge     : Claude 3.7 Sonnet (Anthropic API — est. \$20-30 full run)"

  PAIR_ARGS=(
    --output   datasets/generated_pair.csv
    --max_iter 5
    --seed     42
  )

  if [[ $TEST_MODE -eq 1 ]]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════╗"
    echo "  ║  TEST MODE: 3 PAIR conversations, verbose ON         ║"
    echo "  ╚══════════════════════════════════════════════════════╝"
    PAIR_ARGS+=(--target 3 --verbose)
  else
    PAIR_ARGS+=(--target 200)
  fi

  if [[ $APPEND_MODE -eq 1 ]]; then
    echo "  INFO: Append/resume mode enabled for PAIR generation."
    PAIR_ARGS+=(--append)
  fi

  python project/scripts/generate_pair.py "${PAIR_ARGS[@]}"
  echo "✓ Phase 2g done."
fi

# ── Step 2: Assemble master dataset ──────────────────────────────────────────
echo ""
echo "── Step 2: Assembling master dataset ──────────────────────"
python project/scripts/build_master_csv.py \
  --balance \
  --max_total 3000 \
  --seed 42
echo "✓ master.csv, train.csv, and submission_dataset.csv assembled."

# ── Step 3: Gap analysis ──────────────────────────────────────────────────────
echo ""
echo "── Step 3: Coverage gap analysis ──────────────────────────"
python project/scripts/gap_analysis.py --min_per_signal 50

# ── Step 4: Phase 3 gap-fill (if needed) ────────────────────────────────────
if [[ -f "datasets/gap_signals.txt" ]]; then
  DEFICIT_SIGNALS=$(cat datasets/gap_signals.txt)
  if [[ -n "$DEFICIT_SIGNALS" ]]; then
    echo ""
    echo "── Step 4: Phase 3 — Gap-fill for: $DEFICIT_SIGNALS ─────"

    GAPFILL_ARGS=(
      --signals    "$DEFICIT_SIGNALS"
      --per_signal 40
      --low_risk   0
      --output     datasets/generated_gapfill.csv
      --seed       99
    )

    if [[ $TEST_MODE -eq 1 ]]; then
      echo ""
      echo "  ╔══════════════════════════════════════════════════════╗"
      echo "  ║  TEST MODE: 2 per deficit signal, verbose ON         ║"
      echo "  ╚══════════════════════════════════════════════════════╝"
      # Override per_signal and add verbose
      GAPFILL_ARGS=(
        --signals    "$DEFICIT_SIGNALS"
        --per_signal 2
        --low_risk   0
        --output     datasets/generated_gapfill.csv
        --seed       99
        --verbose
      )
    fi

    python project/scripts/generate_scratch.py "${GAPFILL_ARGS[@]}"
    echo "✓ Gap-fill done."

    # ── Step 5: Re-assemble ───────────────────────────────────────────────────
    echo ""
    echo "── Step 5: Re-assembling master dataset ────────────────"
    python project/scripts/build_master_csv.py \
      --balance \
      --max_total 3500 \
      --seed 42
    echo "✓ Re-assembly done."
  else
    echo "  No coverage gaps detected — skipping Phase 3 gap-fill."
  fi
else
  echo "  No gap_signals.txt found — all signals meet coverage threshold."
fi

echo ""
echo "============================================================"
echo " Assembly complete — $(date)"
echo " Outputs:"
echo "   datasets/master.csv"
echo "   datasets/train.csv"
echo "   datasets/submission_dataset.csv"
echo "============================================================"
