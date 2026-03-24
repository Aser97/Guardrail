"""
fix_softened_rows.py
Correctional script — Phase 2d signal_soften rows in augmented.csv.

Problem: augment_seed.py Phase 2d (signal softening) had no external quality gate.
The softening LLM may have over-softened some conversations, producing rows that are
labeled high-risk but where the signal has effectively disappeared. These are the most
dangerous mislabeled rows in the dataset: a confident high-risk label on content that
a careful reader would find low-risk.

Fix: for every row with source=signal_soften, call Claude Sonnet to score the presence
of the active signals (those labeled 1 in the row's signal columns). If the maximum
score across all active signals is < THRESHOLD, the row is removed.
The file is updated in-place.

Run AFTER making a backup of augmented.csv.

Usage:
    python project/scripts/fix_softened_rows.py \
        --input     datasets/augmented.csv \
        [--threshold 5] \
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

This conversation is labeled as containing the following safety signal(s):
{ACTIVE_SIGNAL_DEFS}

Read the conversation carefully. For each listed signal, score how detectable it is from 0 to 10:
  0   = completely absent — no evidence of this signal anywhere
  1–4 = faint traces, easily dismissed
  5–6 = present but ambiguous — borderline
  7–10 = clearly detectable by a trained safety expert

Conversation:
{TEXT}

Output JSON only (score each active signal 0–10):
{TEMPLATE}
"""


def _get_active_signals(row: dict) -> list[str]:
    return [s for s in SIGNALS if int(row.get(s, 0) or 0) == 1]


def _score_signals(client: AnthropicClient, text: str, active: list[str]) -> dict[str, int]:
    defs = "\n".join(f"  {s}: {SIGNAL_DESCRIPTIONS[s]}" for s in active)
    template = json.dumps({s: 0 for s in active}, indent=2)
    prompt = _JUDGE_PROMPT.format(
        ACTIVE_SIGNAL_DEFS=defs,
        TEXT=text,
        TEMPLATE=template,
    )
    raw = client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    ).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Unparseable judge response: {raw[:200]}")
    data = json.loads(raw[start:end])
    return {s: int(data.get(s, 0)) for s in active}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True, help="Path to augmented.csv")
    p.add_argument("--threshold", type=int, default=5,
                   help="Rows where max active signal score < threshold are removed (default: 5)")
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

    soften_rows = [r for r in rows if r.get("source") == "signal_soften"]
    other_rows  = [r for r in rows if r.get("source") != "signal_soften"]
    LOGGER.info("signal_soften rows to scan: %d  |  Other rows (untouched): %d",
                len(soften_rows), len(other_rows))

    kept, removed = [], []

    for i, row in enumerate(soften_rows):
        text = row.get("text", "")
        active = _get_active_signals(row)

        if not text:
            LOGGER.warning("Row %d has no text — keeping.", i)
            kept.append(row)
            continue

        if not active:
            # Row claims to be high-risk but no signal columns are set — remove it
            LOGGER.warning("Row %d REMOVED — labeled high-risk but no active signal columns.", i)
            removed.append(row)
            continue

        try:
            scores = _score_signals(client, text, active)
            max_score = max(scores.values())
            if max_score < args.threshold:
                LOGGER.warning(
                    "Row %d REMOVED — signal disappeared after softening. "
                    "Active signals %s scored: %s (max=%d)",
                    i, active, scores, max_score,
                )
                removed.append(row)
            else:
                kept.append(row)
                LOGGER.info("Row %d OK  active=%s  max_score=%d", i, active, max_score)
        except Exception as exc:
            LOGGER.warning("Row %d judge error (%s) — keeping row", i, exc)
            kept.append(row)

    LOGGER.info("Result: %d kept, %d removed out of %d signal_soften rows.",
                len(kept), len(removed), len(soften_rows))

    if args.dry_run:
        LOGGER.info("DRY RUN — file not modified.")
        return 0

    final_rows = other_rows + kept
    tmp = input_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    tmp.replace(input_path)

    LOGGER.info("✓ %s updated — %d total rows (%d over-softened rows removed).",
                input_path, len(final_rows), len(removed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
