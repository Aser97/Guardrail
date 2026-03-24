"""
fix_signal_annotation.py
Correctional script — CAMEL and PAIR co-occurring signal annotation.

Problem: generated_camel.csv and generated_camel_hard.csv annotate all 9 signal
columns using Mistral (the same model that generated the user turns). Mistral has
systematic biases when annotating signals it just produced: it may over-annotate
co-occurring signals it intended to embed, or miss signals that emerged organically.
The same issue is assumed to affect generated_pair.csv and generated_pair_hard.csv.

Fix: re-annotate all 9 signal columns in the target CSV using Claude Sonnet. The
primary_signal column (the seeded signal, confirmed present by the CAMEL/PAIR judge)
is forced to 1 regardless of Sonnet's annotation — the judge already confirmed its
presence at generation time with full context. Only co-occurring signals are updated
from Sonnet's re-annotation. The `label` binary field is also recomputed.

The file is updated in-place.

Run AFTER making a backup of the target CSV.
Run this script separately for each CSV that needs re-annotation:
    - datasets/generated_camel.csv
    - datasets/generated_camel_hard.csv
    - datasets/generated_pair.csv
    - datasets/generated_pair_hard.csv

Usage:
    python project/scripts/fix_signal_annotation.py \
        --input   datasets/generated_camel.csv \
        [--dry_run]

    python project/scripts/fix_signal_annotation.py \
        --input   datasets/generated_pair.csv \
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

# Sources that are intentionally label=0 (adversarial negatives).
# primary_signal for these rows is the signal they superficially resemble —
# NOT one that is genuinely present. Their label must stay 0 regardless of
# what Sonnet annotates.
_ADVERSARIAL_SOURCES = {
    "adversarial_negative",
    "pair_adversarial_negative",
    "pair_hard_adversarial_negative",
}

_ANNOTATION_PROMPT = """\
You are a clinical annotation expert for a youth mental health safety classifier.

Read this conversation and label the presence (1) or absence (0) of each safety signal.
A signal is PRESENT (1) only if it appears in user turns with meaningful intensity —
not incidental, metaphorical, or one-word occurrences a careful reader would dismiss.

Signal definitions:
{SIGNAL_DEFS}

Conversation:
{TEXT}

Output JSON only (0 or 1 for each signal, no explanation):
{TEMPLATE}
"""


def _annotate(client: AnthropicClient, text: str) -> dict[str, int]:
    sig_defs = "\n".join(f"  {s}: {SIGNAL_DESCRIPTIONS[s]}" for s in SIGNALS)
    template = json.dumps({s: 0 for s in SIGNALS}, indent=2)
    prompt = _ANNOTATION_PROMPT.format(
        SIGNAL_DEFS=sig_defs,
        TEXT=text,
        TEMPLATE=template,
    )
    raw = client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    ).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Unparseable annotation response: {raw[:200]}")
    data = json.loads(raw[start:end])
    return {s: int(bool(data.get(s, 0))) for s in SIGNALS}


def _parse_signals_field(raw_value: str) -> dict[str, int]:
    """Parse the `signals` JSON column. Returns dict of signal → 0/1."""
    try:
        data = json.loads(raw_value)
        return {s: int(bool(data.get(s, 0))) for s in SIGNALS}
    except Exception:
        return {s: 0 for s in SIGNALS}


def _signals_to_json(signals: dict[str, int]) -> str:
    return json.dumps({s: signals[s] for s in SIGNALS})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input",   required=True,
                   help="Path to CSV to re-annotate (camel, camel_hard, pair, pair_hard)")
    p.add_argument("--dry_run", action="store_true",
                   help="Log changes without modifying the file")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        LOGGER.error("File not found: %s", input_path)
        return 1

    client = AnthropicClient()

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    LOGGER.info("Loaded %d rows from %s", len(rows), input_path)

    # Detect schema: does this CSV use a `signals` JSON column or individual columns?
    has_signals_column   = "signals" in fieldnames
    has_individual_cols  = any(s in fieldnames for s in SIGNALS)

    if not has_signals_column and not has_individual_cols:
        LOGGER.error("Cannot detect signal schema in %s — aborting.", input_path)
        return 1

    LOGGER.info("Schema detected: %s",
                "signals JSON column" if has_signals_column else "individual signal columns")

    updated_rows = []
    n_changed = 0
    n_errors  = 0

    for i, row in enumerate(rows):
        text = row.get("text", "")
        primary = row.get("primary_signal", "")

        if not text:
            LOGGER.warning("Row %d has no text — skipping re-annotation.", i)
            updated_rows.append(row)
            continue

        # Read the current signals
        if has_signals_column:
            old_signals = _parse_signals_field(row.get("signals", "{}"))
        else:
            old_signals = {s: int(row.get(s, 0) or 0) for s in SIGNALS}

        is_adversarial_negative = row.get("source", "") in _ADVERSARIAL_SOURCES

        try:
            new_signals = _annotate(client, text)

            # Force primary_signal = 1 only for genuine high-risk rows — the
            # generation judge verified signal_presence >= 7 at generation time;
            # a single annotation pass may miss subtle signals.
            # Do NOT apply to adversarial negatives (their label must stay 0).
            if primary and primary in SIGNALS and not is_adversarial_negative:
                if new_signals.get(primary, 0) == 0:
                    LOGGER.info(
                        "Row %d: Sonnet gave 0 for primary signal '%s' — forcing to 1 "
                        "(confirmed by generation judge).", i, primary
                    )
                new_signals[primary] = 1

            # Check if anything changed
            changed = {s for s in SIGNALS if old_signals.get(s, 0) != new_signals.get(s, 0)}
            if changed:
                n_changed += 1
                LOGGER.info("Row %d updated — changed signals: %s", i,
                            {s: (old_signals[s], new_signals[s]) for s in changed})
            else:
                LOGGER.info("Row %d — no change.", i)

            # Write updated signal values back to row
            new_row = dict(row)
            if has_signals_column:
                new_row["signals"] = _signals_to_json(new_signals)
                if is_adversarial_negative:
                    # Label must stay 0 — the generation judge confirmed this is
                    # genuinely low-risk. If Sonnet found a signal, flag it for
                    # manual review but do not corrupt the label.
                    if any(v == 1 for v in new_signals.values()):
                        flagged = [s for s, v in new_signals.items() if v == 1]
                        LOGGER.warning(
                            "Row %d [adversarial_negative] — Sonnet flagged signals %s "
                            "but label kept at 0. Review this row manually.", i, flagged
                        )
                    new_row["label"] = 0
                else:
                    new_row["label"] = int(any(v == 1 for v in new_signals.values()))
            else:
                for s in SIGNALS:
                    new_row[s] = new_signals[s]
                # Update high_risk_any if present
                if "high_risk_any" in new_row:
                    new_row["high_risk_any"] = int(any(v == 1 for v in new_signals.values()))
                if "label" in new_row:
                    new_row["label"] = int(any(v == 1 for v in new_signals.values()))

            updated_rows.append(new_row)

        except Exception as exc:
            LOGGER.warning("Row %d annotation error (%s) — keeping original.", i, exc)
            updated_rows.append(row)
            n_errors += 1

    LOGGER.info("Re-annotation complete: %d/%d rows updated, %d errors.",
                n_changed, len(rows), n_errors)

    if args.dry_run:
        LOGGER.info("DRY RUN — file not modified.")
        return 0

    tmp = input_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    tmp.replace(input_path)

    LOGGER.info("✓ %s updated — %d rows re-annotated, %d changed.",
                input_path, len(updated_rows), n_changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
