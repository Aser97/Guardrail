"""
scripts/generate_camel.py
Phase 1b — CAMEL dual-agent generation.

Two LLMs take turns playing roles in a synthetic conversation:
    USER role    → Mistral Large 3  (voices the person in distress)
    ASSISTANT role → C4AI Command A (voices the AI assistant / support chat)

The CAMEL framework prevents the user-role model from "breaking character"
and produces more naturalistic back-and-forth than asking a single model to
write the whole conversation at once.

Architecture:
    1. A "seed" system prompt is sent to each model (different role framing).
    2. Models alternate: user turn → assistant turn → user turn → ...
    3. After N_TURNS the conversation is collected and annotated.
    4. A lightweight annotation call (GPT-OSS) labels the 9 signals.

Output: datasets/generated_camel.csv

Usage:
    python project/scripts/generate_camel.py \\
        --target 300 \\
        --output datasets/generated_camel.csv \\
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
import re
import random
import sys
import time
from pathlib import Path


from config import (
    SIGNALS, SIGNAL_DESCRIPTIONS,
    ESCALATION_STAGES, ESCALATION_WEIGHTS,
    REGISTER_TYPES, REGISTER_DESCRIPTIONS,
    LANGUAGES, LANGUAGE_NOTES, LANGUAGE_WEIGHTS,
    TEMPERATURE,
    CAMEL_CSV, PERSONA_BANK_PATH, SLANG_BANK_PATH,
    PHRASE_CATALOG,
)
from typing import IO
from llm_client import PrivateClient, AnthropicClient, PRIVATE_MISTRAL, PRIVATE_SUPPORT, CLAUDE_SONNET
from utils import (
    persona_to_text, stressor_to_text,
    init_csv, append_row, count_csv_rows, vprint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Turn counts are sampled per-conversation in the generation loop based on
# escalation stage — full escalation gets more room to develop.
_N_TURNS_STANDARD = (16, 20)
_N_TURNS_FULL     = (22, 28)

# ──────────────────────────────────────────────────────────────────────────────
# System prompts for each role
# ──────────────────────────────────────────────────────────────────────────────

def user_system_prompt(
    signal: str,
    phrase: str,
    escalation_stage: str,
    register: str,
    language: str,
    persona: dict,
    prior_feedback: str = "",
) -> str:
    persona_prose  = persona_to_text(persona)
    stressor_prose = stressor_to_text(persona)
    stressor_block = f"\n\nWhat they are currently going through:\n{stressor_prose}" if stressor_prose else ""
    feedback_block = f"\n\nPrior attempt feedback (signal expression):\n  {prior_feedback}" if prior_feedback else ""
    return f"""You are roleplaying as a real person reaching out to an AI assistant chat.
You are NOT an AI. Here is who you are:

{persona_prose}{stressor_block}

Your role in this conversation:
- Write ONLY your own messages (the person, not the AI).
- Your messages must reflect the signal "{signal}": {SIGNAL_DESCRIPTIONS[signal]}
- Key phrase to express naturally (do not quote verbatim): {phrase}
- Escalation stage: {escalation_stage} — calibrate the intensity accordingly.
- Write in {language} ({LANGUAGE_NOTES[language]}).
- Register: {register} — {REGISTER_DESCRIPTIONS[register]}
- Stay in character. Do NOT break character, add stage directions, or explain yourself.
- Do NOT write the AI's responses — only your own messages.
- KEEP EACH MESSAGE SHORT: 1–3 sentences maximum. No long paragraphs.{feedback_block}
- Start the conversation now with your first message."""


def assistant_system_prompt(register: str, language: str, prior_feedback: str = "") -> str:
    feedback_block = f"\n\nPrior attempt feedback (tone/realism):\n  {prior_feedback}" if prior_feedback else ""
    return f"""You are a compassionate AI mental-health support assistant.
You are having a text conversation with a user who may be in distress.

Your role:
- Respond empathetically, without judgment.
- Keep responses concise (2–3 sentences max).
- Do NOT diagnose, prescribe, or provide clinical advice.
- Do NOT trigger or amplify the user's distress signals.
- Write in {language}.
- Register: {register} — {REGISTER_DESCRIPTIONS[register]}
- Write ONLY your response — do not include the user's message.
- Do NOT break character or add meta-commentary.
- KEEP RESPONSES SHORT: 2–3 sentences maximum.
- Do NOT use emojis, exclamation chains, or terms of endearment (no "sweetie", "hun", etc.).
- You are a professional support chatbot, NOT a friend. Maintain warm but boundaried tone.{feedback_block}"""


_CAMEL_JUDGE_PROMPT = """\
You are evaluating a dual-agent training conversation for a youth mental health safety classifier.

Target signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}

Conversation:
{CONVERSATION_TEXT}

Score on three axes from 1 to 10:

1. signal_presence: Is {SIGNAL_NAME} genuinely detectable in the user's messages?
   (1 = not present at all, 10 = unambiguously present)
   → REQUIRED ≥ 7 to accept

2. realism: Does the user sound like a real youth aged 13–22 in distress? \
Does the assistant maintain warm, professional empathy WITHOUT using emojis, \
acting overly friendly, or behaving like a peer rather than a support service?
   (1 = clearly synthetic or inappropriate tone, 10 = both sides realistic)
   → REQUIRED ≥ 7 to accept

3. subtlety: How indirectly is the signal expressed by the user?
   (1 = named explicitly, 10 = requires inference)
   → INFORMATIONAL ONLY — no pass/fail gate

If rejected, provide separate one-sentence feedback:
- signal_feedback: for the USER role — how to better embed the signal
- support_feedback: for the SUPPORT role — how to fix tone/register/realism

Output JSON only:
{{"signal_presence": X, "realism": X, "subtlety": X, "accept": true/false, \
"signal_feedback": "...", "support_feedback": "..."}}
"""

# Maximum judge retries per conversation (each attempt = full CAMEL run — keep low)
_CAMEL_MAX_JUDGE_RETRIES = 2


def _call_camel_judge(
    judge_client: AnthropicClient,
    conversation: str,
    signal: str,
) -> dict:
    """Call Claude Sonnet to evaluate a CAMEL conversation. Returns parsed JSON dict."""
    prompt = _CAMEL_JUDGE_PROMPT.format(
        SIGNAL_NAME=signal,
        SIGNAL_DEFINITION=SIGNAL_DESCRIPTIONS.get(signal, ""),
        CONVERSATION_TEXT=conversation,
    )
    raw = judge_client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE["annotation"],
        max_tokens=300,
    ).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"CAMEL judge returned unparseable response: {raw[:200]}")
    return json.loads(m.group())


def _camel_judge_accepts(result: dict) -> bool:
    return (
        result.get("accept", False)
        and result.get("signal_presence", 0) >= 7
        and result.get("realism", 0) >= 7
    )


ANNOTATION_PROMPT = """You are a clinical annotation assistant.

Read the following conversation and label the presence (1) or absence (0) of each safety signal.
A signal is PRESENT (1) if it appears at least once in any user message.

SIGNALS to label:
{signal_list}

CONVERSATION:
{conversation}

Return ONLY a valid JSON object with exactly these keys (integer values 0 or 1):
{{
  "burden_language": 0,
  "finality_language": 0,
  "escape_framing": 0,
  "hopelessness": 0,
  "active_self_harm": 0,
  "immediate_safety": 0,
  "self_image_crisis": 0,
  "third_party_concern": 0,
  "testing": 0
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# CAMEL dialogue engine
# ──────────────────────────────────────────────────────────────────────────────

def run_camel_dialogue(
    *,
    client: PrivateClient,
    signal: str,
    phrase: str,
    escalation_stage: str,
    register: str,
    language: str,
    persona: dict,
    n_turns: int,
    user_feedback: str = "",
    support_feedback: str = "",
    verbose: bool = False,
) -> str:
    """
    Run the CAMEL dual-agent loop and return the full conversation as a string.
    Feedback strings (from a prior judge rejection) are injected into system prompts.
    """
    user_sys   = user_system_prompt(signal, phrase, escalation_stage, register, language, persona, prior_feedback=user_feedback)
    asst_sys   = assistant_system_prompt(register, language, prior_feedback=support_feedback)

    vprint(verbose,
        f"\n{'━'*60}",
        f"[1b CAMEL] signal={signal}  stage={escalation_stage}  lang={language}  n_turns={n_turns}",
        f"[USER system prompt]",
        f"{'─'*60}",
        user_sys,
        f"[ASSISTANT system prompt]",
        f"{'─'*60}",
        asst_sys,
        f"{'━'*60}",
    )

    user_history:  list[dict] = [{"role": "system", "content": user_sys}]
    asst_history:  list[dict] = [{"role": "system", "content": asst_sys}]
    conversation_lines: list[str] = []

    for turn in range(n_turns):
        # ── User turn (Mistral Large via Mistral AI API) ─────────────────────
        user_msg = client.complete(
            model=PRIVATE_MISTRAL,
            messages=user_history,
            temperature=TEMPERATURE["user_turn"],
            max_tokens=120,
        ).strip()

        if not user_msg:
            LOGGER.warning("Empty user turn at turn %d — stopping dialogue.", turn)
            break

        vprint(verbose, f"  [turn {turn+1} USER]  {user_msg}")
        conversation_lines.append(f"User: {user_msg}")
        user_history.append({"role": "assistant", "content": user_msg})
        asst_history.append({"role": "user",      "content": user_msg})

        # ── Assistant turn (Llama 70B via Together AI) ──────────────────────
        asst_msg = client.complete(
            model=PRIVATE_SUPPORT,
            messages=asst_history,
            temperature=TEMPERATURE["assistant"],
            max_tokens=120,
        ).strip()

        if not asst_msg:
            LOGGER.warning("Empty assistant turn at turn %d — stopping dialogue.", turn)
            break

        vprint(verbose, f"  [turn {turn+1} SUPP]  {asst_msg}")
        conversation_lines.append(f"Support: {asst_msg}")
        asst_history.append({"role": "assistant", "content": asst_msg})
        user_history.append({"role": "user",      "content": asst_msg})

    return "\n".join(conversation_lines)


def annotate_signals(client: PrivateClient, conversation: str) -> dict[str, int]:
    """Use Mistral Large (via Mistral AI API) to label the 9 signals on a finished conversation."""
    signal_list = "\n".join(
        f"  {sig}: {SIGNAL_DESCRIPTIONS[sig]}" for sig in SIGNALS
    )
    prompt = ANNOTATION_PROMPT.format(
        signal_list=signal_list,
        conversation=conversation,
    )
    try:
        raw = client.complete(
            model=PRIVATE_MISTRAL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE["annotation"],
            max_tokens=300,
        )
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        data  = json.loads(raw[start:end])
        return {s: int(data.get(s, 0)) for s in SIGNALS}
    except Exception as exc:
        LOGGER.warning("Annotation failed: %s — defaulting to zeros.", exc)
        return {s: 0 for s in SIGNALS}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def weighted_choice(options: list[str], weights: dict[str, float], rng: random.Random) -> str:
    return rng.choices(list(options), weights=[weights.get(o, 1.0) for o in options], k=1)[0]


FIELDNAMES = [
    "text", "label", "source", "primary_signal",
    "escalation_stage", "register", "language", "persona_id", "signals",
]


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d rows to %s", len(rows), path)


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def generate(
    *,
    client: PrivateClient,
    judge_client: AnthropicClient,
    target: int,
    persona_bank: list[dict],
    rng: random.Random,
    writer: "csv.DictWriter[str]",
    fh: "IO[str]",
    dry_run: bool = False,
    verbose: bool = False,
    existing_count: int = 0,
) -> int:
    """Generate CAMEL conversations and write each row immediately.

    Returns number of rows written this run.
    """
    remaining = max(0, target - existing_count)
    if remaining == 0:
        LOGGER.info("CAMEL: already at %d/%d — skipping.", existing_count, target)
        return 0
    LOGGER.info("CAMEL: need %d more (have %d/%d).", remaining, existing_count, target)

    n_written = 0
    n_failed  = 0
    i         = 0

    while n_written < remaining:
        entry    = rng.choice(PHRASE_CATALOG)
        signal   = entry["signal"]
        phrase   = entry["phrase"]
        category = entry["category"]
        stage    = weighted_choice(ESCALATION_STAGES, ESCALATION_WEIGHTS, rng)
        register = rng.choice(REGISTER_TYPES)
        language = weighted_choice(LANGUAGES, LANGUAGE_WEIGHTS, rng)
        persona  = rng.choice(persona_bank)
        n_turns  = rng.randint(*_N_TURNS_FULL) if stage == "full" else rng.randint(*_N_TURNS_STANDARD)

        if dry_run:
            conversation = f"[DRY RUN CAMEL] signal={signal} phrase={phrase} stage={stage}"
            signal_dict  = {s: (1 if s == signal else 0) for s in SIGNALS}
        else:
            try:
                conversation = None
                user_feedback: str = ""
                support_feedback: str = ""

                for judge_attempt in range(_CAMEL_MAX_JUDGE_RETRIES + 1):
                    try:
                        conv_candidate = run_camel_dialogue(
                            client=client,
                            signal=signal,
                            phrase=phrase,
                            escalation_stage=stage,
                            register=register,
                            language=language,
                            persona=persona,
                            n_turns=n_turns,
                            user_feedback=user_feedback,
                            support_feedback=support_feedback,
                            verbose=verbose,
                        )
                    except Exception as exc:
                        LOGGER.warning("CAMEL dialogue error (judge attempt %d): %s", judge_attempt, exc)
                        break

                    try:
                        judge_result = _call_camel_judge(judge_client, conv_candidate, signal)
                        vprint(verbose,
                            f"[1b JUDGE attempt {judge_attempt+1}]",
                            f"  signal_presence={judge_result.get('signal_presence')}  "
                            f"realism={judge_result.get('realism')}  "
                            f"subtlety={judge_result.get('subtlety')}  "
                            f"accept={judge_result.get('accept')}",
                        )
                    except Exception as exc:
                        LOGGER.warning("CAMEL judge error (attempt %d): %s — accepting anyway.", judge_attempt, exc)
                        conversation = conv_candidate
                        break

                    if _camel_judge_accepts(judge_result):
                        conversation = conv_candidate
                        break
                    else:
                        user_feedback    = judge_result.get("signal_feedback", "")
                        support_feedback = judge_result.get("support_feedback", "")
                        LOGGER.info(
                            "  judge rejected (attempt %d/%d) — signal=%d realism=%d",
                            judge_attempt + 1, _CAMEL_MAX_JUDGE_RETRIES + 1,
                            judge_result.get("signal_presence", 0),
                            judge_result.get("realism", 0),
                        )

                if conversation is None:
                    raise RuntimeError(f"All {_CAMEL_MAX_JUDGE_RETRIES + 1} judge attempts failed or dialogue errored.")

                signal_dict = annotate_signals(client, conversation)
                signal_dict[signal] = 1
            except Exception as exc:
                n_failed += 1
                LOGGER.warning("CAMEL conversation failed (row %d): %s", i, exc)
                time.sleep(2)
                if n_failed > remaining // 2:
                    LOGGER.error("Too many failures — stopping.")
                    break
                continue

        high_risk_any = int(any(v == 1 for v in signal_dict.values()))
        row = {
            "text":             conversation,
            "label":            high_risk_any,
            "source":           "generate_camel",
            "primary_signal":   signal,
            "escalation_stage": stage,
            "register":         register,
            "language":         language,
            "persona_id":       persona.get("id", ""),
            "signals":          json.dumps(signal_dict),
            "category":         category,
        }
        append_row(writer, fh, row)
        n_written += 1
        i         += 1

        if i % 25 == 0:
            LOGGER.info("Progress: %d / %d conversations this run.", n_written, remaining)

    LOGGER.info("CAMEL generation done: %d rows this run, %d failed.", n_written, n_failed)
    return n_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1b: CAMEL dual-agent generation")
    parser.add_argument("--target",   type=int, default=300,       help="Target conversation count")
    parser.add_argument("--output",   default=str(CAMEL_CSV),      help="Output CSV path")
    parser.add_argument("--seed",     type=int, default=42,        help="Random seed")
    parser.add_argument("--dry_run",  action="store_true",         help="Simulate without API calls")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print every turn of every CAMEL dialogue to stdout.")
    parser.add_argument("--append",   action="store_true",
                        help="Append to existing output; skip conversations already written.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)

    existing_count = count_csv_rows(output_path) if args.append else 0
    if existing_count:
        LOGGER.info("Append mode: found %d existing rows in %s.", existing_count, output_path)

    if not PERSONA_BANK_PATH.exists():
        LOGGER.error("Persona bank not found — run build_persona_bank.py first.")
        return 1
    with open(PERSONA_BANK_PATH, "r", encoding="utf-8") as f:
        persona_bank: list[dict] = json.load(f)
    LOGGER.info("Loaded %d personas.", len(persona_bank))

    client       = PrivateClient()
    judge_client = AnthropicClient()
    fh, writer = init_csv(output_path, FIELDNAMES, append=args.append)
    try:
        n = generate(
            client=client,
            judge_client=judge_client,
            target=args.target,
            persona_bank=persona_bank,
            rng=rng,
            writer=writer,
            fh=fh,
            dry_run=args.dry_run,
            verbose=args.verbose,
            existing_count=existing_count,
        )
    finally:
        fh.close()

    LOGGER.info("Done. Wrote %d rows this run → %s", n, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
