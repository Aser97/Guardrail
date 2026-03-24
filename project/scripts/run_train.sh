#!/usr/bin/env bash
# scripts/run_train.sh
# Phase 6 — Train Qwen2.5-7B-Instruct + LoRA guardrail classifier
#
# Architecture : Qwen2.5-7B-Instruct + LoRA (r=16, alpha=32)
# Labels       : 9-head multi-label (BCEWithLogitsLoss)
# Hardware     : A6000 48 GB — bf16, no quantization
# Est. runtime : 3-4 hours on A6000
#
# Must be run AFTER run_assemble.sh has produced datasets/train.csv.
#
# Usage:
#   bash scripts/run_train.sh              # full training run (3 epochs)
#   bash scripts/run_train.sh --test       # quick sanity run (1 epoch, small batch)
#   bash scripts/run_train.sh --resume     # resume from checkpoint_last/ after a crash
#
# No API env vars required (training is fully local).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Parse flags ───────────────────────────────────────────────────────────────
TEST_MODE=0
RESUME_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --test)   TEST_MODE=1 ;;
    --resume) RESUME_FLAG="--resume" ;;
    *)
      echo "ERROR: Unknown argument: $arg"
      echo "       Usage: $0 [--test] [--resume]"
      exit 1
      ;;
  esac
done

# ── Install / verify Python dependencies ─────────────────────────────────────
# Paperspace Gradient loses pip packages on notebook shutdown, so we reinstall
# every time. pip is fast if the packages are already up to date.
echo "Installing Python dependencies…"
pip install --quiet peft transformers accelerate --break-system-packages
# flash-attn requires nvcc which is absent on Paperspace Gradient — use PyTorch's
# built-in SDPA (scaled_dot_product_attention) instead, which gives the same
# Flash Attention 2 kernels without any compilation step.

# ── Check prerequisites ───────────────────────────────────────────────────────
if [[ ! -f "datasets/train.csv" ]]; then
  echo "ERROR: datasets/train.csv not found."
  echo "       Run run_assemble.sh first to build the training dataset."
  exit 1
fi

# Warn if no GPU
if ! python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null; then
  echo "WARN: CUDA not available — training will be extremely slow on CPU."
  echo "      If this is intentional (e.g., testing on a CPU machine), continue."
  echo "      Otherwise, ensure you are running on a GPU node (e.g., Paperspace A6000)."
fi

# ── Build training arguments ──────────────────────────────────────────────────
TRAIN_ARGS=(
  --base_model      "Qwen/Qwen2.5-7B-Instruct"
  --output_dir      "project/models/mhs_guardrail"
  --dtype           bfloat16
  --device          cuda
  --lora_r          16
  --lora_alpha      32
  --lora_dropout    0.05
  --lora_targets    "q_proj,k_proj,v_proj,o_proj"
  --max_length      4096
  --threshold       0.5
  --test_fraction   0.15
  --seed            42
)

if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  TEST MODE: 1 epoch, batch_size=2, grad_accum=2          ║"
  echo "║  Purpose: verify forward/backward pass and checkpoint     ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  TRAIN_ARGS+=(
    --data       "datasets/train.csv"
    --epochs     1
    --batch_size 2
    --grad_accum 2
    --lr         2e-4
    --warmup_ratio 0.05
  )
else
  TRAIN_ARGS+=(
    --data       "datasets/train.csv"
    --epochs     3
    --batch_size 4
    --grad_accum 4
    --lr         2e-4
    --warmup_ratio 0.05
  )
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 6 — Training Qwen2.5-7B-Instruct + LoRA guardrail"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"

ROW_COUNT=$(python -c "
import csv
with open('datasets/train.csv') as f:
    print(sum(1 for _ in csv.DictReader(f)))
" 2>/dev/null || echo "unknown")
echo "  Training rows: $ROW_COUNT"
echo "  Output dir   : project/models/mhs_guardrail/"

mkdir -p project/models/mhs_guardrail

python project/scripts/train_qwen_guardrail.py "${TRAIN_ARGS[@]}" $RESUME_FLAG

echo ""
echo "✓ Training complete — $(date)"
echo "  Model artefacts: project/models/mhs_guardrail/"
echo ""
echo "  Next step: bash scripts/run_verify.sh"
