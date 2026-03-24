"""
scripts/evolve_conversations.py
Phase 2f — Evol-Instruct complexity evolution + Constitutional AI self-check.

Applies three evolution operators to existing conversations:
  Deepen     — add emotional depth, more layered escalation, turn-by-turn build
  Diversify  — rewrite with different vocabulary / register / code-switching pattern
  Complicate — introduce a confounding low-risk narrative alongside the high-risk signal

Each evolved conversation inherits the parent's signal labels.
A Constitutional AI-style self-check asks the same model to verify the evolved version
still contains the intended signal — rejecting it if not. This replaces an external
judge call, keeping this phase free (hackathon endpoint only).

Usage:
    python project/scripts/evolve_conversations.py \\
        --inputs  datasets/generated_scratch.csv datasets/generated_camel.csv \\
        --output  datasets/evolved.csv \\
        --target  120 \\
        --seed    42
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging
import random
import re
import sys
import uuid
from pathlib import Path


from typing import IO
from config import SIGNALS, SIGNAL_DESCRIPTIONS, TEMPERATURE
from llm_client import PrivateClient, PRIVATE_MISTRAL, PRIVATE_SUPPORT
from utils import init_csv, append_row, count_csv_rows, vprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Model aliases for readability within this script
MISTRAL = PRIVATE_MISTRAL   # Mistral AI API — creative rewriting + self-check
SUPPORT = PRIVATE_SUPPORT   # Together AI — diversify role

# ──────────────────────────────────────────────────────────────────────────────
# Operator prompts
# ──────────────────────────────────────────────────────────────────────────────

_DEEPEN_PROMPT = """\
You are evolving a training conversation for a youth mental health safety classifier.

OPERATOR: DEEPEN
Add more emotional layers to the existing conversation while preserving its core narrative.
Specifically:
- Add more turn-by-turn build-up before the signal peaks (make the escalation feel earned)
- Include more emotional contradictions (the user downplays, then reveals more)
- Add at least one moment where the user deflects or backtracks before continuing
- Preserve all existing signals and their labels

Active signals in this conversation: {ACTIVE_SIGNALS}
Language: {LANGUAGE}

Original conversation:
{CONVERSATION_TEXT}

Assistant turns must remain professional: warm and empathetic, but boundaried. No emojis, no exclamation chains, no terms of endearment. 2–3 sentences per assistant turn.
Output: deepened conversation only (same user:/assistant: format). No explanation.
"""

_DIVERSIFY_PROMPT = """\
You are evolving a training conversation for a youth mental health safety classifier.

OPERATOR: DIVERSIFY
Rewrite the following conversation with a noticeably different vocabulary and register,
while preserving all risk signals and their emotional content.
Specifically:
- Use different slang terms, sentence structures, and code-switching patterns
- Change the opening topic or framing if possible (same underlying distress, different entry point)
- Do NOT change which signals are present or their intensity
- Language: {LANGUAGE}

Active signals: {ACTIVE_SIGNALS}

Original conversation:
{CONVERSATION_TEXT}

Assistant turns must remain professional: warm and empathetic, but boundaried. No emojis, no exclamation chains, no terms of endearment. 2–3 sentences per assistant turn.
Output: diversified conversation only (same user:/assistant: format). No explanation.
"""

_COMPLICATE_PROMPT = """\
You are evolving a training conversation for a youth mental health safety classifier.

OPERATOR: COMPLICATE
Introduce a confounding low-risk narrative thread alongside the existing high-risk signal.
This tests whether the classifier can detect the signal even when it's surrounded by
mundane, non-risky content.
Specifically:
- Add a subplot: the user is also discussing something completely ordinary
  (homework, a game, a show, a friend drama) throughout the conversation
- The high-risk signal must still emerge naturally — it is NOT drowned out
- The classifier must still be able to detect: {ACTIVE_SIGNALS}
- Do NOT add new high-risk signals

Language: {LANGUAGE}

Original conversation:
{CONVERSATION_TEXT}

Assistant turns must remain professional: warm and empathetic, but boundaried. No emojis, no exclamation chains, no terms of endearment. 2–3 sentences per assistant turn.
Output: complicated conversation only (same user:/assistant: format). No explanation.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Constitutional AI self-check prompt
# ──────────────────────────────────────────────────────────────────────────────

_SELF_CHECK_PROMPT = """\
You evolved a youth chatbot conversation. Now verify your work.

Expected signals (must be present): {ACTIVE_SIGNALS}
Each signal definition:
{SIGNAL_DEFS}

Evolved conversation:
{EVOLVED_TEXT}

For each expected signal, answer: is it still detectable in this conversation? (yes/no)
Then give a final verdict: ACCEPT or REJECT.
ACCEPT only if ALL expected signals are still detectable.

Output JSON:
{{
  "signal_checks": {{"signal_name": "yes/no", ...}},
  "verdict": "ACCEPT" or "REJECT",
  "reason": "one sentence"
}}
"""


def _self_check(
    evolved_text: str,
    active_signals: list[str],
    client: PrivateClient,
    model: str,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Ask the model to verify its own evolution. Returns (accepted, reason)."""
    sig_defs = "\n".join(
        f"  {s}: {SIGNAL_DESCRIPTIONS.get(s, '')}" for s in active_signals
    )
    prompt = _SELF_CHECK_PROMPT.format(
        ACTIVE_SIGNALS=", ".join(active_signals),
        SIGNAL_DEFS=sig_defs,
        EVOLVED_TEXT=evolved_text,
    )
    vprint(verbose,
        f"[2f SELF-CHECK  model={model}]",
        f"{'─'*60}",
        prompt,
        f"{'─'*60}",
    )
    try:
        raw = client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE["annotation"],
            max_tokens=400,
        ).strip()
        vprint(verbose, f"[SELF-CHECK RESPONSE]  {raw}")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return False, "could not parse self-check response"
        result = json.loads(m.group())
        accepted = result.get("verdict", "REJECT").upper() == "ACCEPT"
        reason = result.get("reason", "")
        return accepted, reason
    except Exception as exc:
        return False, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Per-row evolution
# ──────────────────────────────────────────────────────────────────────────────

_OPERATORS = ["deepen", "diversify", "complicate"]
_OPERATOR_PROMPTS = {
    "deepen":     _DEEPEN_PROMPT,
    "diversify":  _DIVERSIFY_PROMPT,
    "complicate": _COMPLICATE_PROMPT,
}
# Which model to use per operator (complicate requires more creativity)
_OPERATOR_MODEL = {
    "deepen":     MISTRAL,
    "diversify":  SUPPORT,
    "complicate": MISTRAL,
}


def _get_active_signals(row: dict) -> list[str]:
    return [s for s in SIGNALS if int(row.get(s, 0) or 0) == 1]


def evolve_row(
    row: dict,
    operator: str,
    client: PrivateClient,
    rng: random.Random,
    verbose: bool = False,
) -> dict | None:
    """Apply one evolution operator to a single row. Returns evolved row or None."""
    text = row.get("text", "")
    lang = row.get("language", "en")
    active = _get_active_signals(row)

    if not text or not active:
        return None

    prompt_template = _OPERATOR_PROMPTS[operator]
    prompt = prompt_template.format(
        ACTIVE_SIGNALS=", ".join(active),
        LANGUAGE=lang,
        CONVERSATION_TEXT=text,
    )
    # French conversations always routed to Mistral regardless of operator.
    model = MISTRAL if lang == "fr" else _OPERATOR_MODEL[operator]

    vprint(verbose,
        f"\n{'━'*60}",
        f"[2f EVOLVE  op={operator}  model={model}  lang={lang}  signals={active}]",
        f"{'─'*60}",
        prompt,
        f"{'━'*60}",
    )

    try:
        evolved_text = client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE["user_turn"],
            max_tokens=3000,
        ).strip()
        vprint(verbose,
            f"[EVOLVED TEXT]",
            f"{'─'*60}",
            evolved_text,
            f"{'━'*60}",
        )
    except Exception as exc:
        LOGGER.warning("evolve %s failed: %s", operator, exc)
        return None

    # Constitutional AI self-check
    accepted, reason = _self_check(evolved_text, active, client, MISTRAL, verbose=verbose)
    if not accepted:
        LOGGER.info("  self-check REJECT (%s): %s", operator, reason)
        return None

    new_row = dict(row)
    new_row["text"]   = evolved_text
    new_row["source"] = row.get("source", "unknown") + f"_evol_{operator}"
    # Remove conversation_id — not in master FIELDNAMES, would be dropped by extrasaction=ignore
    new_row.pop("conversation_id", None)
    LOGGER.info("  evolved (%s): signals=%s", operator, active)
    return new_row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 2f: Evol-Instruct complexity evolution + Constitutional AI self-check."
    )
    p.add_argument("--inputs",  nargs="+", required=True,
                   help="Input CSV paths to evolve from")
    p.add_argument("--output",  default="datasets/evolved.csv")
    p.add_argument("--target",  type=int, default=120,
                   help="Target total evolved rows across all operators")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--verbose", action="store_true",
                   help="Print every prompt, evolved text, and self-check result to stdout.")
    p.add_argument("--append",  action="store_true",
                   help="Append to existing output; reduce target by already-written rows.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    client = PrivateClient()
    output_path = Path(args.output)

    existing_count = count_csv_rows(output_path) if args.append else 0
    if existing_count:
        LOGGER.info("Append mode: found %d existing rows → reducing target.", existing_count)
    target = max(0, args.target - existing_count)

    all_rows: list[dict] = []
    for inp in args.inputs:
        rows = load_csv(Path(inp))
        LOGGER.info("Loaded %d rows from %s", len(rows), inp)
        all_rows.extend(rows)

    if not all_rows:
        LOGGER.error("No input rows found.")
        return 1

    candidates = [r for r in all_rows if _get_active_signals(r)]
    LOGGER.info("%d high-risk rows available for evolution.", len(candidates))

    per_operator = max(1, target // len(_OPERATORS))

    # Determine fieldnames from first candidate row (fallback to SIGNALS-based list)
    sample_fields = list(candidates[0].keys()) if candidates else []
    fieldnames = sample_fields or (
        ["text", "label", "source", "primary_signal",
         "escalation_stage", "register", "language", "persona_id", "signals", "category"]
    )

    fh, writer = init_csv(output_path, fieldnames, append=args.append)
    total_written = 0
    try:
        for operator in _OPERATORS:
            LOGGER.info("=== Operator: %s (target %d) ===", operator, per_operator)
            rng.shuffle(candidates)
            op_written = 0
            for row in candidates:
                if op_written >= per_operator:
                    break
                evolved = evolve_row(row, operator, client, rng, verbose=args.verbose)
                if evolved:
                    append_row(writer, fh, evolved)
                    op_written   += 1
                    total_written += 1
            LOGGER.info("  %s: %d/%d rows accepted and written.", operator, op_written, per_operator)
    finally:
        fh.close()

    LOGGER.info("✓ Total evolved rows this run: %d → %s", total_written, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
