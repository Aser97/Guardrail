"""
scripts/reannotate_augmented.py
Re-annotate signal labels for artifact-contaminated rows in augmented.csv.

Background
----------
datasets/augmented.csv was produced by the cleaning script which:
  - Stripped LLM wrapper artifacts from 196/604 rows (cleaned text)
  - Added two tracking columns: needs_reannotation, artifact_type

This script re-annotates those 196 rows using the Claude Sonnet API so that
signal labels reflect the CLEANED text, not the contaminated original.

Usage (in-place — reads entirely into memory before writing, safe):
    python project/scripts/reannotate_augmented.py \
        --input  datasets/augmented.csv \
        --output datasets/augmented.csv

Or with separate output for safety:
    python project/scripts/reannotate_augmented.py \
        --input  datasets/augmented.csv \
        --output datasets/augmented_reannotated.csv

What this script does
---------------------
1. Reads augmented.csv (all 604 rows, already cleaned text).
2. For rows where needs_reannotation=1 (196 rows): calls Claude Sonnet to
   re-annotate the 9 signal labels on the cleaned text.
3. For rows where needs_reannotation=0 (408 rows): keeps existing signal
   annotations untouched.
4. Recomputes high_risk_any = 1 if any signal == 1, else 0, for ALL rows.
5. Writes the final file in the ORIGINAL augmented.csv column format:
   conversation_id, text, language, category, persona, source,
   high_risk_any, burden_language, finality_language, escape_framing,
   hopelessness, active_self_harm, immediate_safety, self_image_crisis,
   third_party_concern, testing
   (The needs_reannotation and artifact_type tracking columns are REMOVED.)

Requirements
------------
    ANTHROPIC_API_KEY in .env at repo root (loaded automatically via config.py)

Checkpointing
-------------
    Progress is saved after every CHECKPOINT_INTERVAL rows to
    <output>.checkpoint.json so the script can be resumed after interruption.
    Delete the checkpoint file to force a full re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging
import time

from config import SIGNALS, SIGNAL_DESCRIPTIONS
from llm_client import AnthropicClient, CLAUDE_SONNET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS: list[str] = [
    "conversation_id", "text", "language", "category", "persona", "source",
    "high_risk_any",
    *SIGNALS,
]

MAX_TOKENS          = 300
CHECKPOINT_INTERVAL = 25   # save progress every N annotated rows

# ── Annotation prompt (same pattern as fix_signal_annotation.py) ──────────────

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
        max_tokens=MAX_TOKENS,
    ).strip()
    # Strip markdown fences if present
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Unparseable annotation response: {raw[:200]}")
    data = json.loads(raw[start:end])
    return {s: int(bool(data.get(s, 0))) for s in SIGNALS}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict[str, dict]:
    """Return {conversation_id: signal_dict} from a saved checkpoint."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path: Path, done: dict[str, dict]) -> None:
    with open(path, "w") as f:
        json.dump(done, f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Re-annotate signals for artifact rows")
    parser.add_argument(
        "--input", required=True,
        help="Path to augmented.csv (with needs_reannotation column)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the final augmented.csv (original column format)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print how many rows need re-annotation without calling the API",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    ckpt_path   = output_path.with_suffix(".checkpoint.json")

    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        return 1

    # ── Load input entirely into memory (safe for in-place output) ────────────
    rows: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    LOGGER.info("Loaded %d rows from %s", len(rows), input_path)

    needs_reannotation = [r for r in rows if r.get("needs_reannotation", "0") == "1"]
    LOGGER.info(
        "Rows to re-annotate: %d  |  clean rows (keep as-is): %d",
        len(needs_reannotation), len(rows) - len(needs_reannotation),
    )

    if args.dry_run:
        print(f"\nDry run: {len(needs_reannotation)} rows need re-annotation.")
        return 0

    # ── Load checkpoint ───────────────────────────────────────────────────────
    done: dict[str, dict] = load_checkpoint(ckpt_path)
    if done:
        LOGGER.info("Resuming from checkpoint: %d rows already annotated.", len(done))

    # ── Init client (ANTHROPIC_API_KEY loaded by config.py via .env) ──────────
    client = AnthropicClient()

    # ── Annotate ──────────────────────────────────────────────────────────────
    to_annotate = [r for r in needs_reannotation if r["conversation_id"] not in done]
    LOGGER.info("%d rows left to annotate (after checkpoint).", len(to_annotate))

    for i, row in enumerate(to_annotate, start=1):
        conv_id = row["conversation_id"]
        text    = row["text"]

        LOGGER.info(
            "[%d/%d] Annotating conversation_id=%s …",
            i, len(to_annotate), conv_id,
        )
        try:
            signals = _annotate(client, text)
        except Exception as exc:
            LOGGER.warning(
                "Giving up on conversation_id=%s — keeping original: %s", conv_id, exc
            )
            signals = {s: int(row.get(s, "0") or "0") for s in SIGNALS}

        done[conv_id] = signals

        if i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(ckpt_path, done)
            LOGGER.info(
                "Checkpoint saved (%d/%d done).", len(done), len(needs_reannotation)
            )

        # Polite pause to avoid hammering the API
        time.sleep(0.3)

    # Final checkpoint save
    save_checkpoint(ckpt_path, done)
    LOGGER.info("All annotations complete. Saving output…")

    # ── Write output (original column format, tracking columns stripped) ──────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_changed_high = 0
    n_changed_low  = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            out = {k: row.get(k, "") for k in OUTPUT_COLUMNS}
            conv_id = row["conversation_id"]

            if conv_id in done:
                new_signals = done[conv_id]
                for s in SIGNALS:
                    out[s] = new_signals[s]
            else:
                for s in SIGNALS:
                    out[s] = int(row.get(s, "0") or "0")

            # Recompute high_risk_any from final signals (enforce consistency)
            high_risk = 1 if any(int(out[s]) == 1 for s in SIGNALS) else 0

            old_high_risk = int(row.get("high_risk_any", "0") or "0")
            if old_high_risk != high_risk:
                if high_risk == 1:
                    n_changed_high += 1
                else:
                    n_changed_low += 1

            out["high_risk_any"] = high_risk
            writer.writerow(out)

    LOGGER.info("Saved %d rows to %s", len(rows), output_path)
    LOGGER.info(
        "Label changes: %d flipped to high_risk=1, %d flipped to high_risk=0",
        n_changed_high, n_changed_low,
    )

    if ckpt_path.exists():
        ckpt_path.unlink()
        LOGGER.info("Checkpoint file removed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
