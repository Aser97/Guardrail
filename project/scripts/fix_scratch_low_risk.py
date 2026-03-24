"""
fix_scratch_low_risk.py
Correctional script — Phase 1a low-risk rows.

Problem: generated_scratch.csv low-risk rows (high_risk_any=0) were generated
without an external quality gate. Some may accidentally contain real safety signals.

Fix: run every low-risk row through Claude Sonnet. If any signal scores >= THRESHOLD,
the row is removed. The file is updated in-place.

Run AFTER making a backup of generated_scratch.csv.

Usage:
    python project/scripts/fix_scratch_low_risk.py \
        --input     datasets/generated_scratch.csv \
        [--threshold 6] \
        [--dry_run]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging
from pathlib import Path

from config import SIGNALS, SIGNAL_DESCRIPTIONS
from llm_client import AnthropicClient, CLAUDE_SONNET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are a safety annotation expert reviewing training data for a youth mental health classifier.

Read this conversation and score the presence of each safety signal from 0 to 10.

Signal definitions:
{SIGNAL_DEFS}

Conversation:
{TEXT}

Score each signal from 0 to 10:
  0   = definitely absent
  1–5 = vague, incidental, or metaphorical — a careful reader would dismiss it
  6–10 = meaningfully present — a trained safety expert would flag this for review

Output JSON only (one integer per signal, no explanation):
{TEMPLATE}
"""


def _score_signals(client: AnthropicClient, text: str) -> dict[str, int]:
    sig_defs = "\n".join(f"  {s}: {SIGNAL_DESCRIPTIONS[s]}" for s in SIGNALS)
    template = json.dumps({s: 0 for s in SIGNALS}, indent=2)
    prompt = _JUDGE_PROMPT.format(SIGNAL_DEFS=sig_defs, TEXT=text, TEMPLATE=template)
    raw = client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    ).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Unparseable judge response: {raw[:200]}")
    data = json.loads(raw[start:end])
    return {s: int(data.get(s, 0)) for s in SIGNALS}


def _is_low_risk(row: dict) -> bool:
    val = row.get("high_risk_any", row.get("label", "0"))
    try:
        return int(val or 0) == 0
    except (ValueError, TypeError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True, help="Path to generated_scratch.csv")
    p.add_argument("--threshold", type=int, default=6,
                   help="Any signal scoring >= threshold triggers removal (default: 6)")
    p.add_argument("--dry_run",   action="store_true",
                   help="Log removals without modifying the file")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        LOGGER.error("File not found: %s", input_path)
        return 1

    client = AnthropicClient()

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    LOGGER.info("Loaded %d rows from %s", len(rows), input_path)

    low_risk  = [r for r in rows if     _is_low_risk(r)]
    high_risk = [r for r in rows if not _is_low_risk(r)]
    LOGGER.info("Low-risk to scan: %d  |  High-risk (untouched): %d",
                len(low_risk), len(high_risk))

    kept, removed = [], []

    for i, row in enumerate(low_risk):
        text = row.get("text", "")
        if not text:
            LOGGER.warning("Row %d has no text — keeping.", i)
            kept.append(row)
            continue
        try:
            scores = _score_signals(client, text)
            flagged = {s: v for s, v in scores.items() if v >= args.threshold}
            if flagged:
                LOGGER.warning("Row %d REMOVED — accidental signals detected: %s",
                               i, flagged)
                removed.append(row)
            else:
                kept.append(row)
                LOGGER.info("Row %d OK  (max score: %d)", i, max(scores.values()))
        except Exception as exc:
            LOGGER.warning("Row %d judge error (%s) — keeping row", i, exc)
            kept.append(row)

    LOGGER.info("Result: %d kept, %d removed out of %d low-risk rows scanned.",
                len(kept), len(removed), len(low_risk))

    if args.dry_run:
        LOGGER.info("DRY RUN — file not modified.")
        return 0

    final_rows = high_risk + kept
    tmp = input_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    tmp.replace(input_path)

    LOGGER.info("✓ %s updated — %d total rows (%d low-risk rows removed).",
                input_path, len(final_rows), len(removed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
