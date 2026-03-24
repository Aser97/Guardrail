"""
fix_augmented_labels.py
Correctional script — Phase 2a/2b augmented rows in augmented.csv.

Problem: augment_seed.py Phases 2a (language rewrite) and 2b (persona swap /
esconv_swapped) inherit signal labels from the parent row without verifying that
the signal survived the transformation. A French rewrite or a persona swap to a
very different age/register can dilute or erase the signal while the label stays
high-risk. This is a silent quality drag: the label is directionally harmless (it
claims high-risk, not low-risk) but trains the classifier on weak or absent signal
expression.

Fix: for every high-risk row with source in {lang_rewrite, persona_swap,
esconv_swapped}, call Claude Sonnet to score the active signals. If the maximum
score across all active signals is < THRESHOLD, the row is removed.
Low-risk augmented rows of these types are not scanned here (their reverse concern
— accidental signal introduction — is out of scope for this script).
The file is updated in-place.

Run AFTER making a backup of augmented.csv.
Run AFTER fix_softened_rows.py (separate concern, separate pass).

Usage:
    python project/scripts/fix_augmented_labels.py \
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

# Sources produced by Phases 2a and 2b
_TARGET_SOURCES = {"lang_rewrite", "persona_swap", "esconv_swapped"}

_JUDGE_PROMPT = """\
You are a safety annotation expert reviewing training data for a youth mental health classifier.

This conversation is labeled as containing the following safety signal(s):
{ACTIVE_SIGNAL_DEFS}

Read the conversation carefully. For each listed signal, score how detectable it is from 0 to 10:
  0   = completely absent
  1–4 = faint traces, easily dismissed by a careful reader
  5–6 = present but ambiguous — borderline
  7–10 = clearly detectable by a trained safety expert

Conversation:
{TEXT}

Output JSON only (score each active signal 0–10):
{TEMPLATE}
"""


def _get_active_signals(row: dict) -> list[str]:
    return [s for s in SIGNALS if int(row.get(s, 0) or 0) == 1]


def _is_high_risk(row: dict) -> bool:
    val = row.get("high_risk_any", row.get("label", "0"))
    try:
        return int(val or 0) == 1
    except (ValueError, TypeError):
        return False


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
                   help="High-risk rows where max active signal score < threshold are removed (default: 5)")
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

    # Rows we scan: target sources AND high-risk
    to_scan  = [r for r in rows if r.get("source") in _TARGET_SOURCES and _is_high_risk(r)]
    to_skip  = [r for r in rows if not (r.get("source") in _TARGET_SOURCES and _is_high_risk(r))]

    LOGGER.info("High-risk 2a/2b rows to scan: %d  |  Other rows (untouched): %d",
                len(to_scan), len(to_skip))

    kept, removed = [], []

    for i, row in enumerate(to_scan):
        text = row.get("text", "")
        active = _get_active_signals(row)
        source = row.get("source", "?")

        if not text:
            LOGGER.warning("Row %d [%s] has no text — keeping.", i, source)
            kept.append(row)
            continue

        if not active:
            # Claims high-risk but no signal columns set — inherited label with no content
            LOGGER.warning("Row %d [%s] REMOVED — high-risk label but no active signal columns.", i, source)
            removed.append(row)
            continue

        try:
            scores = _score_signals(client, text, active)
            max_score = max(scores.values())
            if max_score < args.threshold:
                LOGGER.warning(
                    "Row %d [%s] REMOVED — signal did not survive transformation. "
                    "Active signals %s scored: %s (max=%d)",
                    i, source, active, scores, max_score,
                )
                removed.append(row)
            else:
                kept.append(row)
                LOGGER.info("Row %d [%s] OK  active=%s  max_score=%d",
                            i, source, active, max_score)
        except Exception as exc:
            LOGGER.warning("Row %d [%s] judge error (%s) — keeping row", i, source, exc)
            kept.append(row)

    LOGGER.info("Result: %d kept, %d removed out of %d 2a/2b high-risk rows scanned.",
                len(kept), len(removed), len(to_scan))

    if args.dry_run:
        LOGGER.info("DRY RUN — file not modified.")
        return 0

    final_rows = to_skip + kept
    tmp = input_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    tmp.replace(input_path)

    LOGGER.info("✓ %s updated — %d total rows (%d label-mismatch rows removed).",
                input_path, len(final_rows), len(removed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
