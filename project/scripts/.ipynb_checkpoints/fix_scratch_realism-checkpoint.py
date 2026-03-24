"""
fix_scratch_realism.py
Correctional script — Phase 1a high-risk rows.

Problem: generated_scratch.csv high-risk rows were accepted without a realism gate.
CAMEL and PAIR both require realism >= 7 from Claude Sonnet. Phase 1a had no such
filter, so some rows may be too clinical, formulaic, or AI-sounding to be useful
training data.

Fix: run every high-risk row through Claude Sonnet for a realism score. Rows scoring
< THRESHOLD are removed. Low-risk rows are not touched (handled by fix_scratch_low_risk.py).
The file is updated in-place.

Run AFTER making a backup of generated_scratch.csv.

Usage:
    python project/scripts/fix_scratch_realism.py \
        --input     datasets/generated_scratch.csv \
        [--threshold 5] \
        [--dry_run]

Note: threshold is intentionally set lower than CAMEL/PAIR (which use 7) because
Phase 1a is bulk generation — we want to filter obvious failures, not match the
strict standard of the dual-agent phases. Adjust upward if you want stricter culling.
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

import config  # noqa: F401 — side-effect: loads .env into os.environ
from llm_client import AnthropicClient, CLAUDE_SONNET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are reviewing a synthetic training conversation for a youth mental health classifier.

Score the realism of the USER turns only from 1 to 10:
  1  = clearly AI-generated: clinical vocabulary, perfectly structured sentences,
       unnatural emotional rhythm, sounds like a textbook example
  5  = mixed: some authentic moments but noticeable formulaic or artificial phrasing
  10 = highly realistic: genuine youth voice (13-22), natural emotional rhythm,
       register-appropriate informal language, unpredictable in the way real people are

Focus exclusively on how the user sounds. Ignore the assistant turns.

Conversation:
{TEXT}

Output JSON only:
{{"realism": <integer 1-10>, "reason": "<one sentence>"}}
"""


def _score_realism(client: AnthropicClient, text: str) -> tuple[int, str]:
    prompt = _JUDGE_PROMPT.format(TEXT=text)
    raw = client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    ).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Unparseable judge response: {raw[:200]}")
    data = json.loads(raw[start:end])
    return int(data.get("realism", 5)), data.get("reason", "")


def _is_high_risk(row: dict) -> bool:
    val = row.get("high_risk_any", row.get("label", "0"))
    try:
        return int(val or 0) == 1
    except (ValueError, TypeError):
        return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True, help="Path to generated_scratch.csv")
    p.add_argument("--threshold", type=int, default=5,
                   help="Rows with realism < threshold are removed (default: 5)")
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

    high_risk = [r for r in rows if     _is_high_risk(r)]
    low_risk  = [r for r in rows if not _is_high_risk(r)]
    LOGGER.info("High-risk to scan: %d  |  Low-risk (untouched): %d",
                len(high_risk), len(low_risk))

    kept, removed = [], []

    for i, row in enumerate(high_risk):
        text = row.get("text", "")
        if not text:
            LOGGER.warning("Row %d has no text — keeping.", i)
            kept.append(row)
            continue
        try:
            score, reason = _score_realism(client, text)
            if score < args.threshold:
                LOGGER.warning("Row %d REMOVED — realism=%d: %s", i, score, reason)
                removed.append(row)
            else:
                kept.append(row)
                LOGGER.info("Row %d OK  realism=%d", i, score)
        except Exception as exc:
            LOGGER.warning("Row %d judge error (%s) — keeping row", i, exc)
            kept.append(row)

    LOGGER.info("Result: %d kept, %d removed out of %d high-risk rows scanned.",
                len(kept), len(removed), len(high_risk))

    if args.dry_run:
        LOGGER.info("DRY RUN — file not modified.")
        return 0

    final_rows = low_risk + kept
    tmp = input_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    tmp.replace(input_path)

    LOGGER.info("✓ %s updated — %d total rows (%d unrealistic high-risk rows removed).",
                input_path, len(final_rows), len(removed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
