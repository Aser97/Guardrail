"""
scripts/gap_analysis.py
Inspect datasets/master.csv and report coverage gaps:
  – signals with fewer than MIN_PER_SIGNAL samples
  – under-represented languages
  – escalation stage distribution
  – cluster coverage

Run after build_master_csv.py to decide whether to generate more data.

Usage:
    python project/scripts/gap_analysis.py [--min_per_signal 40]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path


from config import SIGNALS, TAXONOMY_CATEGORIES, CONVERSATIONS_PER_CATEGORY, MASTER_CSV

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def analyse(path: Path, min_per_signal: int = 40) -> list[str]:
    if not path.exists():
        print(f"master.csv not found at {path}. Run build_master_csv.py first.")
        return []

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    high_risk = sum(1 for r in rows if str(r.get("label","0")) == "1")

    print(f"\n{'═'*55}")
    print(f"  GAP ANALYSIS — {path.name}")
    print(f"{'═'*55}")
    print(f"  Total rows : {total}")
    print(f"  High-risk  : {high_risk}  ({100*high_risk/max(1,total):.1f}%)")
    print(f"  Low-risk   : {total-high_risk}  ({100*(total-high_risk)/max(1,total):.1f}%)")

    # Signal coverage
    signal_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        sig_str = row.get("signals", "{}")
        try:
            sig_dict = json.loads(sig_str)
        except Exception:
            continue
        for sig in SIGNALS:
            if int(sig_dict.get(sig, 0)) == 1:
                signal_counts[sig] += 1

    print(f"\n  Signal coverage (min target: {min_per_signal}):")
    gaps: list[str] = []
    for sig in SIGNALS:
        count = signal_counts[sig]
        flag  = " ← GAP" if count < min_per_signal else ""
        print(f"    {sig:<30} {count:>4}{flag}")
        if flag:
            gaps.append(sig)

    # Language distribution
    lang_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        lang_counts[row.get("language", "unknown")] += 1
    print(f"\n  Language distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:<15} {count:>5}  ({100*count/max(1,total):.1f}%)")

    # Escalation stage distribution (high-risk only)
    stage_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if str(row.get("label","0")) == "1":
            stage_counts[row.get("escalation_stage","unknown")] += 1
    print(f"\n  Escalation stages (high-risk rows):")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"    {stage:<15} {count:>5}")

    # Register distribution
    reg_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        reg_counts[row.get("register","unknown")] += 1
    print(f"\n  Register distribution:")
    for reg, count in sorted(reg_counts.items(), key=lambda x: -x[1]):
        print(f"    {reg:<35} {count:>5}")

    # Category coverage
    category_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        cat = row.get("category", "").strip()
        if cat:
            category_counts[cat] += 1
    min_per_cat = CONVERSATIONS_PER_CATEGORY // 2   # flag if below half the target
    print(f"\n  Category coverage (target: {CONVERSATIONS_PER_CATEGORY}, flag threshold: {min_per_cat}):")
    cat_gaps: list[str] = []
    for cat in TAXONOMY_CATEGORIES:
        count = category_counts.get(cat, 0)
        flag  = " ← GAP" if count < min_per_cat else ""
        print(f"    {cat:<40} {count:>4}{flag}")
        if flag:
            cat_gaps.append(cat)

    # Recommendations
    if gaps:
        print(f"\n  ⚠  Signals below minimum threshold:")
        for sig in gaps:
            needed = min_per_signal - signal_counts[sig]
            print(f"       {sig:<30} — need ~{needed} more samples")
        print(f"\n     → Re-run generate_scratch.py targeting these signals.")
    else:
        print(f"\n  ✓  All signals meet the minimum threshold of {min_per_signal}.")

    if cat_gaps:
        print(f"\n  ⚠  Categories below half-target ({min_per_cat} conversations):")
        for cat in cat_gaps:
            needed = min_per_cat - category_counts.get(cat, 0)
            print(f"       {cat:<40} — need ~{needed} more samples")
    else:
        print(f"  ✓  All taxonomy categories meet the coverage threshold.")

    print(f"{'═'*55}\n")
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse master.csv for coverage gaps")
    parser.add_argument("--path", default=str(MASTER_CSV), help="Path to master.csv")
    parser.add_argument("--min_per_signal", type=int, default=40, help="Minimum samples per signal")
    args = parser.parse_args()
    gaps = analyse(Path(args.path), min_per_signal=args.min_per_signal)

    # Write gap_signals.txt so train_guardrail.sh Step 5b can read it
    gap_file = Path(args.path).parent / "gap_signals.txt"
    if gaps:
        gap_file.write_text(" ".join(gaps), encoding="utf-8")
        LOGGER.info("Wrote %d deficit signal(s) to %s", len(gaps), gap_file)
    else:
        # Remove stale file so the shell script gap-fill block is skipped
        if gap_file.exists():
            gap_file.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
