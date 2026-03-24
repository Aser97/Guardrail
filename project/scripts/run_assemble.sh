#!/usr/bin/env bash
# scripts/run_assemble.sh
# Assemble all generated CSVs into master.csv, train.csv, and submission_dataset.csv.
#
# Run this AFTER all generation scripts have completed:
#   run_1a.sh, run_1b.sh, run_1b_hard.sh, run_2abcd.sh, run_2g.sh, run_2g_hard.sh
#
# Sources assembled (whichever exist in datasets/):
#   Phase 1a  : datasets/generated_scratch.csv
#   Phase 1b  : datasets/generated_camel.csv
#   Phase 1b-h: datasets/generated_camel_hard.csv
#   ESConv    : datasets/esconv_preprocessed.csv
#   Phase 2a-d: datasets/augmented.csv
#   Phase 2g  : datasets/generated_pair.csv
#   Phase 2g-h: datasets/generated_pair_hard.csv
#
# Note: Phase 2e and Phase 2f are excluded. See data_generation_strategy.md.
#
# Usage:
#   bash scripts/run_assemble.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo " Assembly pipeline"
echo " Repo: $REPO_ROOT"
echo " $(date)"
echo "============================================================"

echo ""
echo "── Assembling master dataset ───────────────────────────────"
echo "   Sources: scratch + camel + camel_hard + esconv + augmented"
echo "            + pair + pair_hard"
echo "   Seed validation rows are automatically excluded."

python project/scripts/build_master_csv.py \
  --balance \
  --max_total 4000 \
  --seed 42

echo ""
echo "============================================================"
echo " Assembly complete — $(date)"
echo " Outputs:"
echo "   datasets/master.csv"
echo "   datasets/train.csv"
echo "   datasets/submission_dataset.csv"
echo "============================================================"
