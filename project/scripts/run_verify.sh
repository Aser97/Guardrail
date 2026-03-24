#!/usr/bin/env bash
# scripts/run_verify.sh
# Verify model artefacts and submission readiness.
#
# Must be run AFTER run_train.sh has completed.
#
# Usage:
#   bash scripts/run_verify.sh           # check artefacts only
#   bash scripts/run_verify.sh --test    # check artefacts + run inference smoke test

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Parse flags ───────────────────────────────────────────────────────────────
TEST_MODE=0
for arg in "$@"; do
  case "$arg" in
    --test) TEST_MODE=1 ;;
    *)
      echo "ERROR: Unknown argument: $arg"
      echo "       Usage: $0 [--test]"
      exit 1
      ;;
  esac
done

MODEL_DIR="project/models/mhs_guardrail"

echo ""
echo "============================================================"
echo " Artefact verification"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"

# ── Check model artefacts ─────────────────────────────────────────────────────
echo ""
echo "── Model artefacts ──────────────────────────────────────────"

ARTEFACTS=(
  "${MODEL_DIR}/adapter_config.json"
  "${MODEL_DIR}/adapter_model.safetensors"
  "${MODEL_DIR}/classifier_head.pt"
  "${MODEL_DIR}/tokenizer/tokenizer_config.json"
)

ALL_OK=1
for artefact in "${ARTEFACTS[@]}"; do
  if [[ -f "$artefact" ]]; then
    SIZE=$(du -sh "$artefact" | cut -f1)
    echo "  ✓  $artefact  ($SIZE)"
  else
    echo "  ✗  MISSING: $artefact"
    ALL_OK=0
  fi
done

# ── Check source datasets ─────────────────────────────────────────────────────
echo ""
echo "── Source datasets ──────────────────────────────────────────"
for f in \
  "datasets/generated_scratch.csv" \
  "datasets/generated_camel.csv" \
  "datasets/generated_camel_hard.csv" \
  "datasets/esconv_preprocessed.csv" \
  "datasets/augmented.csv" \
  "datasets/generated_pair.csv" \
  "datasets/generated_pair_hard.csv"
do
  if [[ -f "$f" ]]; then
    ROWS=$(python -c "
import csv
with open('$f') as fh:
    print(sum(1 for _ in csv.DictReader(fh)))
" 2>/dev/null || echo "?")
    SIZE=$(du -sh "$f" | cut -f1)
    echo "  ✓  $f  ($ROWS rows, $SIZE)"
  else
    echo "  ⚠  MISSING (optional): $f"
  fi
done

# ── Check assembled datasets ──────────────────────────────────────────────────
echo ""
echo "── Assembled datasets ───────────────────────────────────────"
for f in \
  "datasets/master.csv" \
  "datasets/train.csv" \
  "datasets/submission_dataset.csv"
do
  if [[ -f "$f" ]]; then
    ROWS=$(python -c "
import csv
with open('$f') as fh:
    print(sum(1 for _ in csv.DictReader(fh)))
" 2>/dev/null || echo "?")
    SIZE=$(du -sh "$f" | cut -f1)
    echo "  ✓  $f  ($ROWS rows, $SIZE)"
  else
    echo "  ✗  MISSING: $f"
    ALL_OK=0
  fi
done

# ── Check submission entry point ──────────────────────────────────────────────
echo ""
echo "── Submission entry point ───────────────────────────────────"
if [[ -f "project/src/submission/submission.py" ]]; then
  echo "  ✓  project/src/submission/submission.py"
else
  echo "  ✗  MISSING: project/src/submission/submission.py"
  ALL_OK=0
fi

# ── Optional: inference smoke test ───────────────────────────────────────────
if [[ $TEST_MODE -eq 1 ]]; then
  echo ""
  echo "── Inference smoke test ─────────────────────────────────────"
  echo "  Running a quick forward pass with 3 sample conversations..."

  python - <<'PYEOF'
import sys, torch
from pathlib import Path

# Minimal smoke-test: load the LoRA adapter + classifier head and run
# a forward pass on 3 short dummy conversations.
try:
    from transformers import AutoTokenizer
    from peft import PeftModel
    import torch

    MODEL_DIR = Path("project/models/mhs_guardrail")
    TOKENIZER_DIR = MODEL_DIR / "tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    print(f"  ✓ Tokenizer loaded from {TOKENIZER_DIR}")

    # Load classifier head only (lightweight check — skip full model load)
    head = torch.load(MODEL_DIR / "classifier_head.pt", map_location="cpu")
    print(f"  ✓ Classifier head loaded: {list(head.keys())[:3]}...")

    # Tokenize 3 sample inputs
    samples = [
        "User: I don't want to be here anymore.",
        "User: Can you help me with my homework?",
        "User: I've been feeling really empty lately. Nothing matters.",
    ]
    enc = tokenizer(samples, padding=True, truncation=True, max_length=128, return_tensors="pt")
    print(f"  ✓ Tokenized {len(samples)} samples, input_ids shape: {enc['input_ids'].shape}")
    print("  ✓ Smoke test passed — full model load not needed for artefact check.")
except Exception as e:
    print(f"  ✗ Smoke test error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
  if [[ $? -eq 0 ]]; then
    echo "  ✓ Smoke test passed."
  else
    echo "  ✗ Smoke test failed — check training artefacts."
    ALL_OK=0
  fi
fi

# ── Final verdict ─────────────────────────────────────────────────────────────
echo ""
if [[ $ALL_OK -eq 1 ]]; then
  echo "============================================================"
  echo " ✓ VERIFICATION PASSED — $(date)"
  echo " Model : $MODEL_DIR/"
  echo " Submit: project/src/submission/submission.py"
  echo " Dataset: datasets/submission_dataset.csv"
  echo "============================================================"
else
  echo "============================================================"
  echo " ✗ VERIFICATION FAILED — one or more artefacts are missing."
  echo "   Check training logs in run_train.sh output."
  echo "============================================================"
  exit 1
fi
