"""
scripts/augment_seed.py
Phase 2 (types 2a–2d) — LLM-based augmentation of seed conversations.

  2a. Language Rewrite     — EN→FR, EN→mix, FR→EN  (preserves labels)
  2b. Persona Swap         — same scenario + signals, new youth persona + register
  2c. Signal Injection     — low-risk row → inject one high-risk signal in last 4-6 turns
  2d. Signal Softening     — high-risk row → soften signal phrasing to borderline

Reads from:
  --seed_csv     datasets/seed_validation_set.csv   (KHP hackathon seed, binary label)
  --esconv_csv   datasets/esconv_preprocessed.csv   (optional ESConv rows, low-risk)
  --persona_bank datasets/persona_bank.json

Writes to:
  --output       datasets/augmented.csv   (master CSV format)

Notes:
  - ESConv rows are MANDATORY for 2b (persona swap must happen before signal inject).
    They are skipped for 2c/2d until they have been persona-swapped (source=esconv_swapped).
  - All output rows use the internal master CSV schema (with 9 signal columns).
  - For seed rows that only have a binary `label`, we initialize all 9 signal columns to
    label/9 (i.e. if label=1 we set all signals=1; if label=0 all signals=0). This is
    a conservative approximation — the actual per-signal breakdown is unknown for seed rows.

Usage:
    python project/scripts/augment_seed.py \\
        --seed_csv      datasets/seed_validation_set.csv \\
        --esconv_csv    datasets/esconv_preprocessed.csv \\
        --persona_bank  datasets/persona_bank.json \\
        --output        datasets/augmented.csv \\
        --per_type      80 \\
        --seed          42
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
from itertools import cycle
import re
import sys
import uuid
from pathlib import Path
from typing import Optional


from config import (
    SIGNALS,
    PHRASE_CATALOG,
    TEMPERATURE,
    LANGUAGE_NOTES,
)
from typing import IO
from llm_client import PrivateClient, PRIVATE_MISTRAL, PRIVATE_HAIKU
from utils import stressor_to_text, count_csv_rows, vprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

MISTRAL = PRIVATE_MISTRAL  # Mistral Large via Mistral AI API
HAIKU   = PRIVATE_HAIKU    # Claude Haiku via Anthropic API

# French → Mistral (better French quality); English/mixed → round-robin Mistral/Haiku.
_AUG_ROTATION = cycle([MISTRAL, HAIKU])


def _model_for_lang(language: str) -> str:
    """French → Mistral always; English/mixed → round-robin Mistral/Haiku."""
    if language == "fr":
        return MISTRAL
    return next(_AUG_ROTATION)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_row_id() -> str:
    return str(uuid.uuid4())[:8]


def _signals_from_binary_label(label: int) -> dict[str, int]:
    """Fallback when only a binary label is available (seed rows)."""
    v = int(label)
    return {sig: v for sig in SIGNALS}


def _parse_signals(row: dict) -> dict[str, int]:
    """Extract per-signal dict from a row (master CSV or seed CSV)."""
    # Try individual columns first
    if "burden_language" in row:
        return {sig: int(row.get(sig, 0) or 0) for sig in SIGNALS}
    # Fallback to binary label
    label = int(row.get("label", row.get("high_risk_any", 0)) or 0)
    return _signals_from_binary_label(label)


def _signals_to_row_fields(signals: dict[str, int]) -> dict[str, int]:
    return {sig: signals.get(sig, 0) for sig in SIGNALS}


def _seed_row_to_master(row: dict, signals: dict[str, int], source: str) -> dict:
    """Convert a seed/ESConv row to master CSV format."""
    return {
        "conversation_id": row.get("Conversation_id", row.get("conversation_id", _make_row_id())),
        "text":            row.get("Text", row.get("text", "")),
        "language":        row.get("language", "en"),
        "category":        row.get("Category", row.get("category", row.get("taxonomy_category", ""))),
        "persona":         row.get("persona", ""),
        "source":          source,
        "high_risk_any":   int(any(v == 1 for v in signals.values())),
        **_signals_to_row_fields(signals),
    }


# Module-level verbose flag — set in main() before calling augmentation functions
_VERBOSE: bool = False


def _call_llm(client: LLMClient, prompt: str, model: str, temperature: float,
               label: str = "") -> str:
    """Single LLM call, returns text content. Prints prompt+response if verbose."""
    vprint(_VERBOSE,
        f"\n{'━'*60}",
        f"[2abcd {label or 'LLM'} → {model}  T={temperature}]",
        f"{'─'*60}",
        prompt,
        f"{'━'*60}",
    )
    result = client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=3000,
    ).strip()
    vprint(_VERBOSE,
        f"[RESPONSE]",
        f"{'─'*60}",
        result,
        f"{'━'*60}\n",
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2a — Language Rewrite
# ──────────────────────────────────────────────────────────────────────────────

_LANG_LABELS = {
    "en":  "Canadian / North-American English",
    "fr":  "Québec French (informal register: 'faque', 'j'en peux pu', 'checker', 'tsé')",
    "mix": "natural code-switching between English and Québec French mid-message",
}

_LANG_REWRITES: dict[str, list[str]] = {
    "en":  ["fr", "mix"],
    "fr":  ["en"],
    "mix": ["en"],
}

LANG_REWRITE_PROMPT = """\
Rewrite the following youth mental health chatbot conversation entirely in {TARGET_LANG_LABEL}.
Preserve the exact same meaning, risk level, emotional content, and signal labels.
Keep the same number of turns and the same `user: / assistant:` format.
Use authentic {TARGET_LANG_LABEL} youth register — slang, abbreviations, casual spelling.
Do NOT change the category, signals, or risk level.

Original conversation:
{ORIGINAL_TEXT}

Output: rewritten conversation only. No explanation, no preamble.
"""


def augment_language_rewrite(
    rows: list[dict],
    client: LLMClient,
    rng: random.Random,
    per_type: int,
) -> list[dict]:
    """2a: rewrite conversations in a different language."""
    candidates = [r for r in rows if r.get("text")]
    rng.shuffle(candidates)
    results = []
    for row in candidates:
        if len(results) >= per_type:
            break
        src_lang = row.get("language", "en")
        targets = _LANG_REWRITES.get(src_lang, ["fr"])
        tgt_lang = rng.choice(targets)
        prompt = LANG_REWRITE_PROMPT.format(
            TARGET_LANG_LABEL=_LANG_LABELS[tgt_lang],
            ORIGINAL_TEXT=row["text"],
        )
        try:
            model = _model_for_lang(tgt_lang)  # French output → Mistral
            new_text = _call_llm(client, prompt, model, TEMPERATURE["assistant"])
            signals = _parse_signals(row)
            new_row = _seed_row_to_master(row, signals, source="lang_rewrite")
            new_row["conversation_id"] = _make_row_id()
            new_row["text"] = new_text
            new_row["language"] = tgt_lang
            results.append(new_row)
            LOGGER.info("2a rewrite: %s→%s  model=%s  id=%s", src_lang, tgt_lang, model, new_row["conversation_id"])
        except Exception as exc:
            LOGGER.warning("2a failed: %s", exc)
    LOGGER.info("2a language rewrite: %d rows generated.", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2b — Persona Swap
# ──────────────────────────────────────────────────────────────────────────────

PERSONA_SWAP_PROMPT = """\
Rewrite the following chatbot conversation so that the user matches a new persona, \
while keeping the same emotional topic, risk level, and all signal labels intact.

New persona:
{PERSONA_SKETCH}
{STRESSOR_LINE}

Speech style for user turns: {REGISTER_INSTRUCTION}
User turns must NOT use clinical vocabulary or formal sentences.
{ESCONV_NOTE}

Original conversation:
{ORIGINAL_TEXT}

Output: full rewritten conversation only (same `user: / assistant:` format). No explanation.
"""

_REGISTER_BY_AGE: dict[str, str] = {
    "7-12":  "Very simple vocabulary, concrete descriptions, no abbreviations, occasional spelling errors.",
    "13-16": "Heavy slang/abbreviations (lol, idk, ngl, fr, tbh), inconsistent caps, run-on sentences, trailing '...'",
    "17-22": "Mixed register: slang alongside more articulate phrasing; affectively flat delivery.",
    "23-33": "More coherent but still informal; minimizing language ('it's fine', 'nvm'); understatement.",
}


def _get_register(persona: dict) -> str:
    age_group = persona.get("age_group", "17-22")
    return _REGISTER_BY_AGE.get(age_group, _REGISTER_BY_AGE["17-22"])


def augment_persona_swap(
    rows: list[dict],
    personas: list[dict],
    client: LLMClient,
    rng: random.Random,
    per_type: int,
    is_esconv: bool = False,
) -> list[dict]:
    """2b: swap user persona while preserving scenario and signals."""
    candidates = [r for r in rows if r.get("text")]
    rng.shuffle(candidates)
    results = []
    for row in candidates:
        if len(results) >= per_type:
            break
        persona = rng.choice(personas)
        sketch = persona.get("sketch", "")
        stressor_text = stressor_to_text(persona)
        stressor_line = f"Stressor context: {stressor_text}" if stressor_text else ""
        register = _get_register(persona)

        esconv_note = ""
        if is_esconv:
            age = persona.get("age", 17)
            esconv_note = (
                f"The original conversation was written by an adult. Rewrite the user turns "
                f"entirely in the voice of a young person aged {age}. Update vocabulary, "
                f"emotional expression, and register accordingly. Preserve the assistant turns "
                f"with only minimal adjustments for coherence."
            )

        prompt = PERSONA_SWAP_PROMPT.format(
            PERSONA_SKETCH=sketch,
            STRESSOR_LINE=stressor_line,
            REGISTER_INSTRUCTION=register,
            ESCONV_NOTE=esconv_note,
            ORIGINAL_TEXT=row["text"],
        )
        try:
            model = _model_for_lang(row.get("language", "en"))  # French source → Mistral
            new_text = _call_llm(client, prompt, model, TEMPERATURE["assistant"])
            signals = _parse_signals(row)
            new_row = _seed_row_to_master(row, signals, source="persona_swap")
            new_row["conversation_id"] = _make_row_id()
            new_row["text"] = new_text
            new_row["persona"] = sketch[:120]
            if is_esconv:
                new_row["source"] = "esconv_swapped"
                # Language may change if persona has non-EN language_background
                lang_bg = persona.get("language_background", "")
                if "french" in lang_bg.lower() or "français" in lang_bg.lower():
                    new_row["language"] = rng.choice(["fr", "mix"])
            results.append(new_row)
            LOGGER.info("2b swap: id=%s  esconv=%s", new_row["conversation_id"], is_esconv)
        except Exception as exc:
            LOGGER.warning("2b failed: %s", exc)
    LOGGER.info("2b persona swap: %d rows generated.", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2c — Signal Injection (low → high)
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_INJECT_PROMPT = """\
Below is a LOW-RISK youth mental health chatbot conversation.
Modify the last 4–6 user turns to naturally introduce the following signal:

Signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}
Key phrase to embed (do NOT quote verbatim — express naturally): {KEY_PHRASE}

Rules:
- Do NOT change the first 10 turns or assistant turns
- The signal must emerge GRADUALLY and feel authentic to the conversation's emotional context
- Do NOT introduce any other new signals
- Use age-appropriate register; avoid clinical vocabulary
- At the end, output a line: SIGNALS_JSON: <json dict with all 9 signals as 0/1>

Original conversation:
{ORIGINAL_TEXT}

Output: full modified conversation (user:/assistant: format), then SIGNALS_JSON on a new line.
"""


def augment_signal_inject(
    rows: list[dict],
    client: LLMClient,
    rng: random.Random,
    per_type: int,
) -> list[dict]:
    """2c: inject a high-risk signal into low-risk conversations."""
    from config import SIGNAL_DESCRIPTIONS
    low_risk = [r for r in rows if int(r.get("high_risk_any", r.get("label", 0)) or 0) == 0]
    rng.shuffle(low_risk)
    results = []
    for row in low_risk:
        if len(results) >= per_type:
            break
        entry = rng.choice(PHRASE_CATALOG)
        signal = entry["signal"]
        phrase = entry["phrase"]
        definition = SIGNAL_DESCRIPTIONS[signal]
        prompt = SIGNAL_INJECT_PROMPT.format(
            SIGNAL_NAME=signal,
            SIGNAL_DEFINITION=definition,
            KEY_PHRASE=phrase,
            ORIGINAL_TEXT=row["text"],
        )
        try:
            # Signal inject requires precise creative rewriting — always use Mistral
            # (strongest model for this task regardless of source language).
            raw = _call_llm(client, prompt, MISTRAL, TEMPERATURE["user_turn"])
            # Parse out conversation and SIGNALS_JSON
            sig_match = re.search(r"SIGNALS_JSON:\s*(\{[^}]+\})", raw, re.DOTALL)
            if sig_match:
                new_signals = json.loads(sig_match.group(1))
                new_text = raw[: sig_match.start()].strip()
            else:
                new_signals = {sig: 0 for sig in SIGNALS}
                new_signals[signal] = 1
                new_text = raw.strip()

            new_row = _seed_row_to_master(row, new_signals, source="signal_inject")
            new_row["conversation_id"] = _make_row_id()
            new_row["text"] = new_text
            new_row["category"] = entry.get("category", row.get("category", ""))
            new_row["high_risk_any"] = 1
            results.append(new_row)
            LOGGER.info("2c inject: signal=%s  id=%s", signal, new_row["conversation_id"])
        except Exception as exc:
            LOGGER.warning("2c failed: %s", exc)
    LOGGER.info("2c signal inject: %d rows generated.", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2d — Signal Softening (high → borderline)
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_SOFTEN_PROMPT = """\
Below is a HIGH-RISK youth mental health chatbot conversation.
Soften the signal language so it becomes more ambiguous and borderline — a trained safety \
expert would still detect it, but a layperson or keyword filter might miss it.

Rules:
- Preserve the overall emotional tone and narrative arc
- Replace explicit signal phrases with indirect, euphemistic, or metaphorical equivalents
- Do NOT remove the signal entirely — the conversation should still be labelled high-risk
- Do NOT change assistant turns
- Preserve all other signals that are present; only soften the expression, not the meaning

Signals currently present: {ACTIVE_SIGNALS}

Original conversation:
{ORIGINAL_TEXT}

Output: full softened conversation only (user:/assistant: format). No explanation.
"""


def augment_signal_soften(
    rows: list[dict],
    client: LLMClient,
    rng: random.Random,
    per_type: int,
) -> list[dict]:
    """2d: soften signal language in high-risk conversations."""
    high_risk = [
        r for r in rows
        if int(r.get("high_risk_any", r.get("label", 0)) or 0) == 1
    ]
    rng.shuffle(high_risk)
    results = []
    for row in high_risk:
        if len(results) >= per_type:
            break
        signals = _parse_signals(row)
        active = [s for s, v in signals.items() if v == 1]
        if not active:
            continue
        prompt = SIGNAL_SOFTEN_PROMPT.format(
            ACTIVE_SIGNALS=", ".join(active),
            ORIGINAL_TEXT=row["text"],
        )
        try:
            model = _model_for_lang(row.get("language", "en"))  # French → Mistral
            new_text = _call_llm(client, prompt, model, TEMPERATURE["assistant"])
            new_row = _seed_row_to_master(row, signals, source="signal_soften")
            new_row["conversation_id"] = _make_row_id()
            new_row["text"] = new_text
            results.append(new_row)
            LOGGER.info("2d soften: id=%s  signals=%s", new_row["conversation_id"], active)
        except Exception as exc:
            LOGGER.warning("2d failed: %s", exc)
    LOGGER.info("2d signal soften: %d rows generated.", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path) -> None:
    """Full overwrite save (used for final write when --append is not set)."""
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


def flush_phase(rows: list[dict], path: Path, first_write: bool) -> bool:
    """
    Append *rows* to *path*, writing a header only on first_write.

    Returns new value of first_write (False after first call).
    Call after each augmentation phase so partial runs are not lost.
    """
    if not rows:
        return first_write
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")
    mode = "w" if first_write else "a"
    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("  Flushed %d rows to %s (mode=%s).", len(rows), path, mode)
    return False  # no longer first write


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    global _VERBOSE
    p = argparse.ArgumentParser(
        description="Phase 2a-2d: LLM-based augmentation of seed conversations."
    )
    p.add_argument("--seed_csv",     default="datasets/seed_validation_set.csv")
    p.add_argument("--esconv_csv",   default="datasets/esconv_preprocessed.csv")
    p.add_argument("--persona_bank", default="datasets/persona_bank.json")
    p.add_argument("--output",       default="datasets/augmented.csv")
    p.add_argument("--per_type",     type=int, default=80,
                   help="Target rows per augmentation type (2a/2b/2c/2d)")
    p.add_argument("--esconv_swap_target", type=int, default=300,
                   help="Target persona-swapped ESConv rows")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--verbose",      action="store_true",
                   help="Print every LLM prompt and response to stdout.")
    p.add_argument("--append",       action="store_true",
                   help="Append to existing output; proportionally reduce targets by rows already written.")
    args = p.parse_args(argv)

    _VERBOSE = args.verbose
    rng = random.Random(args.seed)
    client = PrivateClient()
    output_path = Path(args.output)

    # Resume: reduce per_type proportionally if file already has rows
    per_type = args.per_type
    esconv_swap_target = args.esconv_swap_target
    first_write = True  # tracks whether to write header when flushing
    if args.append and output_path.exists():
        existing = count_csv_rows(output_path)
        # Each type contributes roughly per_type rows; scale down target
        n_phases = 6  # 2a, 2b-KHP, 2b-ESConv, 2c-KHP, 2c-ESConv, 2d
        already_per_phase = existing // max(n_phases, 1)
        per_type = max(0, per_type - already_per_phase)
        esconv_swap_target = max(0, esconv_swap_target - already_per_phase)
        LOGGER.info(
            "Append mode: %d existing rows → adjusted per_type=%d esconv_swap=%d",
            existing, per_type, esconv_swap_target,
        )
        first_write = False  # file exists — append, no new header

    # ── Load seed data ─────────────────────────────────────────────────────────
    seed_rows   = load_csv(Path(args.seed_csv))
    esconv_rows = load_csv(Path(args.esconv_csv))
    LOGGER.info("Loaded %d seed rows, %d ESConv rows.", len(seed_rows), len(esconv_rows))

    if not seed_rows:
        LOGGER.error("Seed CSV not found or empty: %s", args.seed_csv)
        return 1

    persona_bank_path = Path(args.persona_bank)
    if not persona_bank_path.exists():
        LOGGER.error("Persona bank not found: %s", args.persona_bank)
        return 1
    with open(persona_bank_path, "r", encoding="utf-8") as f:
        personas: list[dict] = json.load(f)
    LOGGER.info("Loaded %d personas.", len(personas))

    seed_master = [_seed_row_to_master(r, _parse_signals(r), "khp_seed") for r in seed_rows]
    esconv_master = [
        _seed_row_to_master(r, _parse_signals(r), "esconv_seed") for r in esconv_rows
    ] if esconv_rows else []

    total_written = 0

    # ── 2a: Language rewrite ───────────────────────────────────────────────────
    LOGGER.info("=== Phase 2a: Language Rewrite ===")
    rows_2a = augment_language_rewrite(seed_master + esconv_master, client, rng, per_type)
    first_write = flush_phase(rows_2a, output_path, first_write)
    total_written += len(rows_2a)

    # ── 2b: Persona swap — KHP ────────────────────────────────────────────────
    LOGGER.info("=== Phase 2b: Persona Swap (KHP seed) ===")
    rows_2b_khp = augment_persona_swap(seed_master, personas, client, rng, per_type, is_esconv=False)
    first_write = flush_phase(rows_2b_khp, output_path, first_write)
    total_written += len(rows_2b_khp)

    # ── 2b: Persona swap — ESConv (mandatory; keep in memory for 2c) ──────────
    esconv_swapped: list[dict] = []
    if esconv_master and esconv_swap_target > 0:
        LOGGER.info("=== Phase 2b: Persona Swap (ESConv — mandatory) ===")
        esconv_swapped = augment_persona_swap(
            esconv_master, personas, client, rng, esconv_swap_target, is_esconv=True
        )
        first_write = flush_phase(esconv_swapped, output_path, first_write)
        total_written += len(esconv_swapped)

    # ── 2c: Signal inject — KHP low-risk ──────────────────────────────────────
    LOGGER.info("=== Phase 2c: Signal Injection (KHP seed low-risk) ===")
    rows_2c_khp = augment_signal_inject(seed_master, client, rng, per_type)
    first_write = flush_phase(rows_2c_khp, output_path, first_write)
    total_written += len(rows_2c_khp)

    # ── 2c: Signal inject — ESConv post-swap ──────────────────────────────────
    if esconv_swapped:
        LOGGER.info("=== Phase 2c: Signal Injection (ESConv post-swap) ===")
        rows_2c_esconv = augment_signal_inject(
            esconv_swapped, client, rng, min(per_type * 2, 150)
        )
        first_write = flush_phase(rows_2c_esconv, output_path, first_write)
        total_written += len(rows_2c_esconv)

    # ── 2d: Signal softening — KHP high-risk ──────────────────────────────────
    LOGGER.info("=== Phase 2d: Signal Softening (KHP seed high-risk) ===")
    rows_2d = augment_signal_soften(seed_master, client, rng, per_type // 2)
    first_write = flush_phase(rows_2d, output_path, first_write)
    total_written += len(rows_2d)

    LOGGER.info("✓ Total augmented rows this run: %d → %s", total_written, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
