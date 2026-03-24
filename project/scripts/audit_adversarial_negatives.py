"""
audit_adversarial_negatives.py
Read-only audit — no API calls.

Scans already-fixed CAMEL/PAIR CSVs for adversarial negative rows where
fix_signal_annotation.py wrote non-zero signal annotations. These are the
rows that triggered "label kept at 0. Review this row manually." warnings
during the fix run.

Because fix_signal_annotation.py writes Sonnet's signal scores into the
`signals` JSON column even for adversarial rows (only the label is protected),
this script can reconstruct the full list of flagged rows from the CSVs alone.

Usage:
    python project/scripts/audit_adversarial_negatives.py \\
        --input datasets/generated_pair.csv \\
        --input datasets/generated_pair_hard.csv

    # Recommended — saves output to file AND shows it in terminal:
    python project/scripts/audit_adversarial_negatives.py \\
        --input datasets/generated_pair.csv \\
        --input datasets/generated_pair_hard.csv \\
        2>&1 | tee adversarial_audit.log
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging

import config  # noqa: F401 — loads .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

_ADVERSARIAL_SOURCES = {
    "adversarial_negative",
    "pair_adversarial_negative",
    "pair_hard_adversarial_negative",
}

# Signal columns — must match config.SIGNALS order
try:
    from config import SIGNALS
except ImportError:
    SIGNALS = [
        "burden_language", "finality_language", "social_isolation",
        "hopelessness", "access_to_means", "third_party_concern",
        "prior_attempt", "testing", "self_harm_non_suicidal",
    ]


def _parse_signals(row: dict) -> dict[str, int]:
    """Parse signals regardless of schema (JSON column or individual columns)."""
    if "signals" in row:
        try:
            data = json.loads(row["signals"])
            return {s: int(bool(data.get(s, 0))) for s in SIGNALS}
        except Exception:
            return {s: 0 for s in SIGNALS}
    return {s: int(row.get(s, 0) or 0) for s in SIGNALS}


def audit_file(path: Path) -> list[dict]:
    """Return list of flagged row dicts (adversarial negatives with signals > 0)."""
    if not path.exists():
        LOGGER.error("File not found: %s", path)
        return []

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    LOGGER.info("Scanning %s — %d total rows", path.name, len(rows))

    flagged = []
    n_adversarial = 0

    for i, row in enumerate(rows):
        source = row.get("source", "")
        if source not in _ADVERSARIAL_SOURCES:
            continue
        n_adversarial += 1

        signals = _parse_signals(row)
        active = {s: v for s, v in signals.items() if v == 1}
        if not active:
            continue

        flagged.append({
            "file":           path.name,
            "row_index":      i,
            "source":         source,
            "primary_signal": row.get("primary_signal", ""),
            "label":          row.get("label", ""),
            "flagged_signals": list(active.keys()),
            "text_preview":   (row.get("text", "") or "")[:120].replace("\n", " "),
        })

    LOGGER.info(
        "%s — %d adversarial negatives scanned, %d flagged (%.1f%%)",
        path.name, n_adversarial, len(flagged),
        100 * len(flagged) / n_adversarial if n_adversarial else 0,
    )
    return flagged


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Audit adversarial negatives for non-zero signal annotations."
    )
    p.add_argument(
        "--input", required=True, action="append", dest="inputs",
        metavar="PATH",
        help="CSV to audit. Repeat for multiple files.",
    )
    args = p.parse_args(argv)

    all_flagged = []
    for inp in args.inputs:
        all_flagged.extend(audit_file(Path(inp)))

    if not all_flagged:
        LOGGER.info("No flagged rows found across %d file(s). All clear.", len(args.inputs))
        return 0

    LOGGER.warning(
        "══════════════════════════════════════════════════════════════"
    )
    LOGGER.warning("FLAGGED ROWS — adversarial negatives with Sonnet-detected signals")
    LOGGER.warning(
        "══════════════════════════════════════════════════════════════"
    )
    for item in all_flagged:
        LOGGER.warning(
            "[%s] row %d | source=%-32s | primary=%-22s | label=%s | "
            "flagged=%s",
            item["file"], item["row_index"], item["source"],
            item["primary_signal"], item["label"],
            item["flagged_signals"],
        )
        LOGGER.warning("  preview: %s", item["text_preview"])

    LOGGER.warning(
        "══════════════════════════════════════════════════════════════"
    )
    LOGGER.warning(
        "Total: %d flagged row(s). Review each — if signals are genuine, "
        "consider removing the row from training data.",
        len(all_flagged),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
