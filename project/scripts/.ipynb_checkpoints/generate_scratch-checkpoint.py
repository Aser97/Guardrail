"""
scripts/generate_scratch.py
Phase 1a — Multi-model from-scratch generation.

For each (signal, escalation_stage, register, language) combination, this script
sends a generation prompt to one of the hackathon LLMs, rotating between all three
for output diversity. French conversations are always routed to Mistral (best French
capability); English and mixed conversations rotate round-robin across all three models.

Each generated conversation is labelled with:
    - The target signal (high_risk_any = 1)
    - All 9 signal scores (1 for target, 0 for others — LLM may annotate later)
    - escalation_stage, register, language, persona_id, source, model_used

Output: datasets/generated_scratch.csv

Usage:
    python project/scripts/generate_scratch.py \\
        --per_signal 60 \\
        --output datasets/generated_scratch.csv \\
        [--low_risk 100] \\
        [--append] \\
        [--dry_run]

Rows are written and flushed to disk after every single conversation.
Use --append to resume an interrupted run without losing any completed work.
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
import sys
import time
from pathlib import Path
from typing import IO


from config import (
    SIGNALS, SIGNAL_DESCRIPTIONS,
    ESCALATION_STAGES, ESCALATION_DESCRIPTIONS, ESCALATION_WEIGHTS,
    REGISTER_TYPES, REGISTER_DESCRIPTIONS,
    LANGUAGES, LANGUAGE_NOTES, LANGUAGE_WEIGHTS,
    TEMPERATURE,
    SCRATCH_CSV, PERSONA_BANK_PATH, SLANG_BANK_PATH,
    PHRASE_CATALOG, TAXONOMY_CATEGORIES,
)
from itertools import cycle
from typing import Callable
from llm_client import LLMClient, MISTRAL, GPT_OSS
from utils import (
    persona_to_text, stressor_to_text,
    init_csv, append_row, count_csv_rows_by_field, vprint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model rotation
# ──────────────────────────────────────────────────────────────────────────────

# All three hackathon models rotate for English/mixed conversations.
# French is always routed to Mistral (strongest French capability).
_ROTATION_MODELS = [MISTRAL, GPT_OSS]  # COMMAND excluded (504 timeouts)


def _make_rotator() -> Callable[[str], str]:
    """Return a stateful picker: French → Mistral, others → round-robin."""
    _cycle = cycle(_ROTATION_MODELS)

    def pick(language: str) -> str:
        if language == "fr":
            return MISTRAL
        return next(_cycle)

    return pick


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical data-generation assistant for a mental-health AI-safety research project.
Your task is to write realistic synthetic conversations that simulate how a person in distress might
communicate with an AI assistant. These conversations will train a safety classifier.
Write with psychological authenticity. Do NOT moralize, lecture, or insert warnings inside the conversation.
Follow all formatting instructions exactly."""


def _sample_slang(slang_bank: dict, language: str, age: int) -> str:
    """Return a short comma-separated list of slang terms appropriate for the persona."""
    if not slang_bank:
        return "(none available)"
    age_group = _age_to_group(age)
    lang_data = slang_bank.get(language, slang_bank.get("en", {}))
    group_data = lang_data.get(age_group, {})
    terms: list[str] = []
    for domain_terms in group_data.values():
        if isinstance(domain_terms, list):
            terms.extend(domain_terms)
    if not terms:
        return "(none available)"
    sample = random.sample(terms, min(6, len(terms)))
    return ", ".join(sample)


def _age_to_group(age: int) -> str:
    if age <= 12:
        return "7-12"
    if age <= 16:
        return "13-16"
    if age <= 22:
        return "17-22"
    return "23-33"


def build_prompt(
    *,
    signal: str,
    phrase: str,
    escalation_stage: str,
    register: str,
    language: str,
    persona: dict,
    slang_bank: dict,
    n_turns: int = 16,
) -> str:
    """Construct the full generation prompt for one conversation."""

    slang_ref      = _sample_slang(slang_bank, language, persona.get("age", 20))
    persona_prose  = persona_to_text(persona)
    stressor_prose = stressor_to_text(persona)
    stressor_block = f"\n─── CURRENT STRESSORS ──────────────────\n{stressor_prose}" if stressor_prose else ""

    prompt = f"""
══════════════════════════════════════════
GENERATION TASK: Synthetic distress conversation
══════════════════════════════════════════

─── PERSONA ───────────────────────────
{persona_prose}{stressor_block}

─── TARGET SIGNAL ─────────────────────
Signal name  : {signal}
Description  : {SIGNAL_DESCRIPTIONS[signal]}
Key phrase   : {phrase}

─── ESCALATION STAGE ──────────────────
Stage  : {escalation_stage}
What it means: {ESCALATION_DESCRIPTIONS[escalation_stage]}

─── LANGUAGE & REGISTER ───────────────
Language : {language}
Note     : {LANGUAGE_NOTES[language]}
Register : {register}
Format   : {REGISTER_DESCRIPTIONS[register]}

─── SLANG REFERENCE ───────────────────
Contextually appropriate terms (use 1–3 naturally, do NOT force all):
{slang_ref}

─── CONVERSATION REQUIREMENTS ─────────
• Write a realistic conversation between the PERSON and an AI assistant.
• Number of turns: approximately {n_turns} (each user message + AI response = 1 turn).
• The PERSON's messages must reflect the target signal and escalation stage.
• The AI assistant responds supportively but does NOT trigger any additional signals.
• Write the conversation in the specified language ({language}).
• Apply the register style precisely.
• The PERSON's authentic voice must be consistent with the persona above.
• Do NOT insert clinical labels, diagnoses, or meta-commentary into the conversation text.
• The signal can be expressed subtly or directly depending on the escalation stage.
• EVERY turn MUST include both a Person message AND an Assistant response — no monologues.

─── TURN FORMAT (strictly follow this) ──
Each turn must be formatted exactly as:
Person: [person's message]
Assistant: [AI assistant's response]

Example of correct format:
Person: i don't know i just feel like nobody cares
Assistant: That sounds really hard. What's been going on?
Person: everything just piles up and i can't take it anymore
Assistant: I hear you. Can you tell me more about what's been piling up?

─── OUTPUT FORMAT ─────────────────────
Return ONLY a JSON object with the following fields — no preamble, no explanation:

{{
  "conversation": "<full conversation using the Person:/Assistant: format above, one turn per line>",
  "signals": {{
    "burden_language": 0,
    "finality_language": 0,
    "escape_framing": 0,
    "hopelessness": 0,
    "active_self_harm": 0,
    "immediate_safety": 0,
    "self_image_crisis": 0,
    "third_party_concern": 0,
    "testing": 0
  }}
}}

IMPORTANT:
• Each signal value is 0 or 1 (integer).
• Set the target signal ({signal}) to 1.
• Set any OTHER signals that genuinely appear in the conversation to 1 as well
  (co-occurrence is natural — do not force it, but do not suppress it).
• The "conversation" field must be the full text only — no JSON inside it.
══════════════════════════════════════════
""".strip()

    return prompt


def build_low_risk_prompt(
    *,
    register: str,
    language: str,
    persona: dict,
    slang_bank: dict,
    n_turns: int = 16,
) -> str:
    """Prompt for a low-risk (zero-signal) conversation."""
    slang_ref      = _sample_slang(slang_bank, language, persona.get("age", 20))
    persona_prose  = persona_to_text(persona)
    stressor_prose = stressor_to_text(persona)
    stressor_block = f"\n─── CURRENT STRESSORS ──────────────────\n{stressor_prose}" if stressor_prose else ""

    prompt = f"""
══════════════════════════════════════════
GENERATION TASK: Low-risk baseline conversation
══════════════════════════════════════════

─── PERSONA ───────────────────────────
{persona_prose}{stressor_block}

─── LANGUAGE & REGISTER ───────────────
Language : {language}
Note     : {LANGUAGE_NOTES[language]}
Register : {register}
Format   : {REGISTER_DESCRIPTIONS[register]}

─── SLANG REFERENCE ───────────────────
{slang_ref}

─── INSTRUCTIONS ──────────────────────
Write a realistic ~{n_turns}-turn conversation where the PERSON discusses mild,
everyday concerns with an AI assistant. None of the 9 safety signals should be
present. The person may sound stressed, sad, or frustrated — but NOT in crisis.
EVERY turn MUST include both a Person message AND an Assistant response.

─── TURN FORMAT (strictly follow this) ──
Each turn must be formatted exactly as:
Person: [person's message]
Assistant: [AI assistant's response]

─── OUTPUT FORMAT ─────────────────────
Return ONLY valid JSON:

{{
  "conversation": "<full conversation using the Person:/Assistant: format, one turn per line>",
  "signals": {{
    "burden_language": 0,
    "finality_language": 0,
    "escape_framing": 0,
    "hopelessness": 0,
    "active_self_harm": 0,
    "immediate_safety": 0,
    "self_image_crisis": 0,
    "third_party_concern": 0,
    "testing": 0
  }}
}}
══════════════════════════════════════════
""".strip()

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def weighted_choice(options: list[str], weights: dict[str, float], rng: random.Random) -> str:
    population = list(options)
    w = [weights.get(o, 1.0) for o in population]
    return rng.choices(population, weights=w, k=1)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str, dict[str, int]]:
    """
    Extract conversation text and signal dict from the model's JSON response.
    Returns (conversation_text, signals_dict).
    """
    # Try to extract JSON from the response
    raw = raw.strip()
    # Find the outermost {...}
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    json_str = raw[start:end]
    data = json.loads(json_str)
    conversation = data.get("conversation", "").strip()
    signals = {s: int(data.get("signals", {}).get(s, 0)) for s in SIGNALS}
    if not conversation:
        raise ValueError("Empty conversation in response")
    return conversation, signals


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def generate_high_risk(
    *,
    client: LLMClient,
    pick_model: Callable[[str], str],
    per_signal: int,
    persona_bank: list[dict],
    slang_bank: dict,
    rng: random.Random,
    writer: "csv.DictWriter[str]",
    fh: "IO[str]",
    dry_run: bool = False,
    verbose: bool = False,
    target_signals: list[str] | None = None,
    existing_per_signal: dict[str, int] | None = None,
) -> int:
    """Generate high-risk conversations and write each row immediately.

    Parameters
    ----------
    writer / fh      : open CSV writer + file handle (from init_csv)
    target_signals   : if set, only generate for these signals (gap-fill mode)
    existing_per_signal : per-signal row counts already in the output file
                          (used by --append to skip already-satisfied signals)

    Returns
    -------
    Total rows written this run.
    """
    import io as _io  # avoid shadowing outer csv import
    n_written = 0
    signals_to_generate = target_signals if target_signals else SIGNALS
    existing = existing_per_signal or {}

    for signal in signals_to_generate:
        already   = existing.get(signal, 0)
        remaining = max(0, per_signal - already)
        if remaining == 0:
            LOGGER.info("Signal %s: already at %d/%d — skipping.", signal, already, per_signal)
            continue
        LOGGER.info("Signal %s: need %d more (have %d/%d).", signal, remaining, already, per_signal)
        n_generated = 0
        n_failed    = 0

        while n_generated < remaining:
            entry    = rng.choice([e for e in PHRASE_CATALOG if e["signal"] == signal])
            phrase   = entry["phrase"]
            category = entry["category"]
            stage    = weighted_choice(ESCALATION_STAGES, ESCALATION_WEIGHTS, rng)
            register = rng.choice(REGISTER_TYPES)
            language = weighted_choice(LANGUAGES, LANGUAGE_WEIGHTS, rng)
            persona  = rng.choice(persona_bank)
            n_turns  = rng.randint(22, 28) if stage == "full" else rng.randint(16, 20)

            model = pick_model(language)

            user_prompt = build_prompt(
                signal=signal,
                phrase=phrase,
                escalation_stage=stage,
                register=register,
                language=language,
                persona=persona,
                slang_bank=slang_bank,
                n_turns=n_turns,
            )

            vprint(verbose,
                f"\n{'━'*60}",
                f"[1a PROMPT] signal={signal}  stage={stage}  lang={language}  register={register}  model={model}",
                f"{'─'*60}",
                user_prompt,
                f"{'━'*60}",
            )

            if dry_run:
                LOGGER.info("[DRY RUN] Would call %s for signal=%s stage=%s", model, signal, stage)
                conversation = f"[DRY RUN] signal={signal} stage={stage} register={register}"
                signal_dict  = {s: (1 if s == signal else 0) for s in SIGNALS}
            else:
                try:
                    raw = client.complete(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=TEMPERATURE["user_turn"],
                        max_tokens=3000,
                    )
                    vprint(verbose,
                        f"[1a RESPONSE — raw]",
                        f"{'─'*60}",
                        raw,
                        f"{'━'*60}\n",
                    )
                    conversation, signal_dict = parse_response(raw)
                except Exception as exc:
                    n_failed += 1
                    LOGGER.warning("Failed generation (signal=%s, attempt %d): %s", signal, n_failed, exc)
                    if n_failed > remaining:
                        LOGGER.error("Too many failures for signal %s — skipping remaining.", signal)
                        break
                    time.sleep(1)
                    continue

            row = {
                "text":              conversation,
                "label":             1,
                "source":            "generate_scratch",
                "primary_signal":    signal,
                "escalation_stage":  stage,
                "register":          register,
                "language":          language,
                "persona_id":        persona.get("id", ""),
                "signals":           json.dumps(signal_dict),
                "category":          category,
                "model_used":        model,
            }
            append_row(writer, fh, row)
            n_generated += 1
            n_written   += 1

        LOGGER.info("Signal %s: %d generated this run, %d failed.", signal, n_generated, n_failed)

    return n_written


def generate_low_risk(
    *,
    client: LLMClient,
    pick_model: Callable[[str], str],
    count: int,
    persona_bank: list[dict],
    slang_bank: dict,
    rng: random.Random,
    writer: "csv.DictWriter[str]",
    fh: "IO[str]",
    dry_run: bool = False,
    verbose: bool = False,
    existing_count: int = 0,
) -> int:
    """Generate low-risk conversations and write each row immediately.

    Returns number of rows written this run.
    """
    remaining = max(0, count - existing_count)
    if remaining == 0:
        LOGGER.info("Low-risk: already at %d/%d — skipping.", existing_count, count)
        return 0

    LOGGER.info("Low-risk: need %d more (have %d/%d).", remaining, existing_count, count)
    n_written = 0
    n_failed  = 0

    lr_personas = [
        p for p in persona_bank
        if p.get("cluster") in ("low_risk_baseline", "adolescent", "young_adult_student", "working_adult")
    ] or persona_bank

    for i in range(remaining):
        register = rng.choice(REGISTER_TYPES)
        language = weighted_choice(LANGUAGES, LANGUAGE_WEIGHTS, rng)
        persona  = rng.choice(lr_personas)
        category = rng.choice(TAXONOMY_CATEGORIES)
        n_turns  = rng.randint(16, 20)
        model    = pick_model(language)

        user_prompt = build_low_risk_prompt(
            register=register,
            language=language,
            persona=persona,
            slang_bank=slang_bank,
            n_turns=n_turns,
        )

        vprint(verbose,
            f"\n{'━'*60}",
            f"[1a LOW-RISK PROMPT]  lang={language}  register={register}  model={model}",
            f"{'─'*60}",
            user_prompt,
            f"{'━'*60}",
        )

        if dry_run:
            conversation = f"[DRY RUN] low_risk register={register}"
            signal_dict  = {s: 0 for s in SIGNALS}
        else:
            try:
                raw = client.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE["user_turn"],
                    max_tokens=3000,
                )
                vprint(verbose,
                    f"[1a LOW-RISK RESPONSE — raw]",
                    f"{'─'*60}",
                    raw,
                    f"{'━'*60}\n",
                )
                conversation, signal_dict = parse_response(raw)
            except Exception as exc:
                n_failed += 1
                LOGGER.warning("Low-risk generation failed (attempt %d): %s", i, exc)
                time.sleep(1)
                continue

        row = {
            "text":              conversation,
            "label":             0,
            "source":            "generate_scratch",
            "primary_signal":    "none",
            "escalation_stage":  "none",
            "register":          register,
            "language":          language,
            "persona_id":        persona.get("id", ""),
            "signals":           json.dumps(signal_dict),
            "category":          category,
            "model_used":        model,
        }
        append_row(writer, fh, row)
        n_written += 1

    LOGGER.info("Low-risk: %d generated this run, %d failed.", n_written, n_failed)
    return n_written


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "text", "label", "source", "primary_signal",
    "escalation_stage", "register", "language", "persona_id", "signals",
    "category", "model_used",
]


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d rows to %s", len(rows), path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1a: multi-model from-scratch generation")
    parser.add_argument("--per_signal", type=int, default=60, help="Target conversations per signal")
    parser.add_argument("--low_risk",   type=int, default=200, help="Number of low-risk conversations")
    parser.add_argument("--output",     default=str(SCRATCH_CSV), help="Output CSV path")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed")
    parser.add_argument("--dry_run",    action="store_true",   help="Simulate without API calls")
    parser.add_argument(
        "--signals", default="", type=str,
        help=(
            "Space-separated list of signals to generate (gap-fill mode). "
            "Defaults to all signals if not set. "
            "Example: --signals 'burden_language hopelessness'"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every prompt and raw LLM response to stdout (test/debug mode).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help=(
            "Append to existing output file and skip signals/counts already satisfied. "
            "Use when resuming an interrupted run."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)

    # Parse optional signal filter (gap-fill mode)
    target_signals: list[str] | None = None
    if args.signals.strip():
        target_signals = [s.strip() for s in args.signals.split() if s.strip() in SIGNALS]
        invalid = [s.strip() for s in args.signals.split() if s.strip() not in SIGNALS]
        if invalid:
            LOGGER.warning("Unknown signal(s) ignored: %s", invalid)
        LOGGER.info("Gap-fill mode: generating only for signals: %s", target_signals)

    # Resume: count what's already in the output file
    existing_per_signal: dict[str, int] = {}
    existing_low_risk = 0
    if args.append and output_path.exists():
        existing_per_signal = count_csv_rows_by_field(output_path, "primary_signal")
        existing_low_risk   = existing_per_signal.pop("none", 0)
        total_existing = sum(existing_per_signal.values()) + existing_low_risk
        LOGGER.info("Append mode: found %d existing rows in %s.", total_existing, output_path)

    # Load persona bank
    if not PERSONA_BANK_PATH.exists():
        LOGGER.error(
            "Persona bank not found at %s — run scripts/build_persona_bank.py first.", PERSONA_BANK_PATH
        )
        return 1
    with open(PERSONA_BANK_PATH, "r", encoding="utf-8") as f:
        persona_bank: list[dict] = json.load(f)
    LOGGER.info("Loaded %d personas.", len(persona_bank))

    slang_bank: dict = {}
    if SLANG_BANK_PATH.exists():
        with open(SLANG_BANK_PATH, "r", encoding="utf-8") as f:
            slang_bank = json.load(f)
        LOGGER.info("Loaded slang bank.")
    else:
        LOGGER.warning("Slang bank not found at %s — proceeding without slang.", SLANG_BANK_PATH)

    client     = LLMClient()
    pick_model = _make_rotator()
    LOGGER.info(
        "Model rotation: French → Mistral always; English/mixed → round-robin %s",
        [MISTRAL, GPT_OSS],
    )

    # Open output file once — rows are written incrementally inside generation functions
    fh, writer = init_csv(output_path, FIELDNAMES, append=args.append)
    try:
        n_high = generate_high_risk(
            client=client,
            pick_model=pick_model,
            per_signal=args.per_signal,
            persona_bank=persona_bank,
            slang_bank=slang_bank,
            rng=rng,
            writer=writer,
            fh=fh,
            dry_run=args.dry_run,
            verbose=args.verbose,
            target_signals=target_signals,
            existing_per_signal=existing_per_signal,
        )
        n_low = generate_low_risk(
            client=client,
            pick_model=pick_model,
            count=args.low_risk,
            persona_bank=persona_bank,
            slang_bank=slang_bank,
            rng=rng,
            writer=writer,
            fh=fh,
            dry_run=args.dry_run,
            verbose=args.verbose,
            existing_count=existing_low_risk,
        )
    finally:
        fh.close()

    LOGGER.info("Done. Wrote %d high-risk + %d low-risk rows this run → %s", n_high, n_low, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
