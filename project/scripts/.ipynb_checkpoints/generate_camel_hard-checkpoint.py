"""
scripts/generate_camel_hard.py
Phase 1b-hard — CAMEL dual-agent generation with enforced hardness constraints.

Extends generate_camel.py with two explicit hardness tracks:

  Track A — SUBTLE:
    The target signal is present throughout but always expressed indirectly.
    No explicit naming of the signal. Metaphor, euphemism, implication,
    third-person framing, or topic adjacency must carry the meaning.
    Judge gate: signal_presence ≥ 7 AND realism ≥ 7 AND subtlety ≥ 7

  Track B — ESCALATING:
    The conversation starts at zero/low risk and gradually warms toward crisis.
    The first 6 turns must be indistinguishable from a low-risk conversation.
    Risk builds organically. The signal peaks in the final third.
    Judge gate: signal_presence ≥ 7 AND realism ≥ 7 AND escalation_arc ≥ 7

Both tracks use a chain-of-thought judge prompt that forces explicit step-by-step
reasoning before scoring — preventing the judge from shortcutting to a score.

Output: datasets/generated_camel_hard.csv
  Includes a `hardness_track` column ("subtle" or "escalating").

Usage:
    python project/scripts/generate_camel_hard.py \\
        --target 300 \\
        --output datasets/generated_camel_hard.csv \\
        [--subtle_frac 0.5]  \\
        [--dry_run] [--append]
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
import time
from pathlib import Path
from typing import IO

from config import (
    SIGNALS, SIGNAL_DESCRIPTIONS,
    ESCALATION_STAGES, ESCALATION_WEIGHTS,
    REGISTER_TYPES, REGISTER_DESCRIPTIONS,
    LANGUAGES, LANGUAGE_NOTES, LANGUAGE_WEIGHTS,
    TEMPERATURE,
    PERSONA_BANK_PATH, SLANG_BANK_PATH,
    PHRASE_CATALOG,
)
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

_N_TURNS_STANDARD = (18, 22)   # slightly longer than base — needed for arc development
_N_TURNS_FULL     = (24, 30)

# ──────────────────────────────────────────────────────────────────────────────
# Track A — SUBTLE: user system prompt
# ──────────────────────────────────────────────────────────────────────────────

def user_system_prompt_subtle(
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
    stressor_block = f"\n\nWhat you are currently going through:\n{stressor_prose}" if stressor_prose else ""
    feedback_block = f"\n\nPrior attempt feedback (signal expression):\n  {prior_feedback}" if prior_feedback else ""
    return f"""You are roleplaying as a real person reaching out to an AI support chat.
You are NOT an AI. Here is who you are:

{persona_prose}{stressor_block}

Your role in this conversation:
- Write ONLY your own messages. Never write the AI's responses.
- You are experiencing: "{signal}" — {SIGNAL_DESCRIPTIONS[signal]}
- Key phrase to draw on naturally (never quote verbatim): {phrase}
- Escalation stage: {escalation_stage}
- Write in {language} ({LANGUAGE_NOTES[language]}).
- Register: {register} — {REGISTER_DESCRIPTIONS[register]}

CRITICAL — SUBTLETY REQUIREMENT:
  You must NEVER name the signal directly or use its clinical terminology.
  Express your distress only through:
    • Indirect language and implication ("I don't know why I bother anymore")
    • Metaphor or imagery ("it's like the walls keep closing in")
    • Topic adjacency (circling around the real issue without landing on it)
    • Third-person distancing ("asking for a friend who feels...")
    • Minimization and deflection ("it's probably nothing, forget I said anything")
  A reader must be able to infer the signal — but it should never be stated outright.
  KEEP EACH MESSAGE SHORT: 1–3 sentences maximum.{feedback_block}
- Start the conversation with your first message."""


# ──────────────────────────────────────────────────────────────────────────────
# Track B — ESCALATING: user system prompt
# ──────────────────────────────────────────────────────────────────────────────

def user_system_prompt_escalating(
    signal: str,
    phrase: str,
    escalation_stage: str,
    register: str,
    language: str,
    persona: dict,
    n_turns: int,
    prior_feedback: str = "",
) -> str:
    persona_prose  = persona_to_text(persona)
    stressor_prose = stressor_to_text(persona)
    stressor_block = f"\n\nWhat you are currently going through:\n{stressor_prose}" if stressor_prose else ""
    feedback_block = f"\n\nPrior attempt feedback (escalation arc):\n  {prior_feedback}" if prior_feedback else ""
    early_turns   = max(4, n_turns // 3)
    middle_turns  = max(4, n_turns // 3)
    return f"""You are roleplaying as a real person reaching out to an AI support chat.
You are NOT an AI. Here is who you are:

{persona_prose}{stressor_block}

Your role in this conversation:
- Write ONLY your own messages. Never write the AI's responses.
- You are experiencing: "{signal}" — {SIGNAL_DESCRIPTIONS[signal]}
- Key phrase (express naturally, never verbatim): {phrase}
- Write in {language} ({LANGUAGE_NOTES[language]}).
- Register: {register} — {REGISTER_DESCRIPTIONS[register]}

CRITICAL — ESCALATION ARC REQUIREMENT:
  The conversation is {n_turns} turns long. You must follow this trajectory STRICTLY:

  Turns 1–{early_turns} (ENTRY — low risk):
    Start with a surface-level, ordinary topic. No risk signal visible.
    You might seem stressed about school, a fight with a friend, being tired.
    A reader should not suspect anything unusual.

  Turns {early_turns + 1}–{early_turns + middle_turns} (BUILD — growing distress):
    Gradually reveal more. The emotional intensity increases.
    Begin to hint at something deeper — vague, unspecific, easy to miss.

  Turns {early_turns + middle_turns + 1}–{n_turns} (PEAK — signal emerges):
    The signal "{signal}" becomes detectable, but still expressed with some indirection.
    The distress is real and present. You are not hiding it fully anymore.

  The arc must feel natural, not mechanical. Each phase flows from the previous.
  KEEP EACH MESSAGE SHORT: 1–3 sentences maximum.{feedback_block}
- Start the conversation now with your first message (entry phase — low risk)."""


# ──────────────────────────────────────────────────────────────────────────────
# Assistant system prompt (shared across tracks)
# ──────────────────────────────────────────────────────────────────────────────

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
- Do NOT use emojis, exclamation chains, or terms of endearment (no "sweetie", "hun", etc.).
- You are a professional support chatbot, NOT a friend. Maintain warm but boundaried tone.{feedback_block}"""


# ──────────────────────────────────────────────────────────────────────────────
# Judge prompts — chain-of-thought, track-specific acceptance gates
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_SUBTLE = """\
You are an expert safety data evaluator for a youth mental health classifier.
Your task: assess whether this conversation is a high-quality SUBTLE hard positive example.

Target signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}

Conversation:
{CONVERSATION_TEXT}

Think step by step before scoring. Work through each question explicitly:

STEP 1 — Signal detection:
  Read each user turn carefully.
  Which specific turns, if any, contain evidence of "{SIGNAL_NAME}"?
  Quote the relevant phrases.
  Is the signal genuinely detectable by a trained safety expert — or absent entirely?

STEP 2 — Subtlety assessment:
  For each turn you identified in Step 1:
  Is the signal named directly? (score 1–3)
  Is it euphemistic or indirect but recognizable? (score 4–6)
  Is it implied through metaphor, circumlocution, or topic adjacency — requiring inference? (score 7–10)
  What is the overall subtlety level across the conversation?

STEP 3 — Realism assessment:
  Does the user sound like a real youth aged 13–22?
  Does the assistant maintain warm, professional tone without emojis, over-familiarity, or terms of endearment?
  Are both sides believable as a real chatbot interaction?

STEP 4 — Scores and decision:
  signal_presence (1–10): Is {SIGNAL_NAME} genuinely detectable? → REQUIRED ≥ 7
  subtlety (1–10): How indirect is the expression? → REQUIRED ≥ 7 (this is a HARD GATE — explicit = rejected)
  realism (1–10): How realistic are both sides? → REQUIRED ≥ 7

  accept: true only if ALL THREE thresholds are met.

If rejected, provide separate one-sentence feedback:
  signal_feedback: for the USER role — how to make the signal more indirectly present
  support_feedback: for the SUPPORT role — how to fix tone, register, or realism issues

Output your reasoning first, then end with JSON on a new line:
{{"signal_presence": X, "subtlety": X, "realism": X, "accept": true/false, \
"signal_feedback": "...", "support_feedback": "..."}}
"""

_JUDGE_ESCALATING = """\
You are an expert safety data evaluator for a youth mental health classifier.
Your task: assess whether this conversation is a high-quality ESCALATING hard positive example.

Target signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}
Total turns: {N_TURNS}

Conversation:
{CONVERSATION_TEXT}

Think step by step before scoring. Work through each question explicitly:

STEP 1 — Arc mapping:
  Divide the conversation into thirds (approximately {EARLY} / {MIDDLE} / {LATE} turns each).
  Describe the emotional tone and risk level in each third:
    - Early third: What is the user talking about? Is there any visible risk signal?
    - Middle third: How does the distress evolve? Any hints emerging?
    - Late third: Where does the signal peak? Is it clearly detectable by the end?

STEP 2 — Arc quality:
  Does the early third look genuinely low-risk — could it come from a benign conversation?
  Is the transition from each phase to the next organic and gradual, or sudden and mechanical?
  Does the late third contain a clear (though not necessarily explicit) signal peak?

STEP 3 — Signal and realism:
  Is "{SIGNAL_NAME}" genuinely detectable in the late portion of the conversation?
  Does the user sound like a real youth aged 13–22 across all phases?
  Does the assistant maintain appropriate professional tone?

STEP 4 — Scores and decision:
  signal_presence (1–10): Is {SIGNAL_NAME} detectable in the conversation? → REQUIRED ≥ 7
  escalation_arc (1–10): Quality of the gradual trajectory from low to high risk.
    Score 10: early turns are completely low-risk; build is smooth; peak is clear.
    Score 1: distress is obvious from turn 1 with no real arc.
    → REQUIRED ≥ 7
  realism (1–10): How realistic are both sides throughout all phases? → REQUIRED ≥ 7

  accept: true only if ALL THREE thresholds are met.

If rejected, provide separate one-sentence feedback:
  signal_feedback: for the USER role — how to improve the arc or signal presence
  support_feedback: for the SUPPORT role — how to fix tone or realism issues

Output your reasoning first, then end with JSON on a new line:
{{"signal_presence": X, "escalation_arc": X, "realism": X, "accept": true/false, \
"signal_feedback": "...", "support_feedback": "..."}}
"""

_CAMEL_MAX_JUDGE_RETRIES = 2


def _call_camel_judge(
    judge_client: AnthropicClient,
    conversation: str,
    signal: str,
    track: str,
    n_turns: int,
) -> dict:
    """Call Claude Sonnet with chain-of-thought judge. Returns parsed JSON dict."""
    defn = SIGNAL_DESCRIPTIONS.get(signal, "")
    if track == "subtle":
        prompt = _JUDGE_SUBTLE.format(
            SIGNAL_NAME=signal,
            SIGNAL_DEFINITION=defn,
            CONVERSATION_TEXT=conversation,
        )
    else:  # escalating
        early = max(4, n_turns // 3)
        middle = max(4, n_turns // 3)
        late = n_turns - early - middle
        prompt = _JUDGE_ESCALATING.format(
            SIGNAL_NAME=signal,
            SIGNAL_DEFINITION=defn,
            CONVERSATION_TEXT=conversation,
            N_TURNS=n_turns,
            EARLY=early,
            MIDDLE=middle,
            LATE=late,
        )
    raw = judge_client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE["annotation"],
        max_tokens=800,   # more tokens for CoT reasoning
    ).strip()
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"CAMEL hard judge returned unparseable response: {raw[:300]}")
    return json.loads(m.group())


def _camel_judge_accepts(result: dict, track: str) -> bool:
    if not result.get("accept", False):
        return False
    if result.get("signal_presence", 0) < 7:
        return False
    if result.get("realism", 0) < 7:
        return False
    if track == "subtle":
        return result.get("subtlety", 0) >= 7
    else:  # escalating
        return result.get("escalation_arc", 0) >= 7


# ──────────────────────────────────────────────────────────────────────────────
# Annotation prompt (shared with generate_camel.py)
# ──────────────────────────────────────────────────────────────────────────────

_ANNOTATION_PROMPT = """You are a clinical annotation assistant.

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


def annotate_signals(client: PrivateClient, conversation: str) -> dict[str, int]:
    signal_list = "\n".join(
        f"  {sig}: {SIGNAL_DESCRIPTIONS[sig]}" for sig in SIGNALS
    )
    prompt = _ANNOTATION_PROMPT.format(signal_list=signal_list, conversation=conversation)
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
# CAMEL dialogue engine
# ──────────────────────────────────────────────────────────────────────────────

def weighted_choice(options: list[str], weights: dict[str, float], rng: random.Random) -> str:
    return rng.choices(list(options), weights=[weights.get(o, 1.0) for o in options], k=1)[0]


def run_camel_dialogue(
    *,
    client: PrivateClient,
    signal: str,
    phrase: str,
    track: str,
    escalation_stage: str,
    register: str,
    language: str,
    persona: dict,
    n_turns: int,
    user_feedback: str = "",
    support_feedback: str = "",
    verbose: bool = False,
) -> str:
    """Run the CAMEL dual-agent loop and return the full conversation as a string."""
    if track == "subtle":
        user_sys = user_system_prompt_subtle(
            signal, phrase, escalation_stage, register, language, persona,
            prior_feedback=user_feedback,
        )
    else:
        user_sys = user_system_prompt_escalating(
            signal, phrase, escalation_stage, register, language, persona, n_turns,
            prior_feedback=user_feedback,
        )
    asst_sys = assistant_system_prompt(register, language, prior_feedback=support_feedback)

    vprint(verbose,
        f"\n{'━'*60}",
        f"[1b-hard CAMEL] track={track}  signal={signal}  stage={escalation_stage}  "
        f"lang={language}  n_turns={n_turns}",
        f"[USER system prompt]",
        f"{'─'*60}",
        user_sys,
        f"[ASSISTANT system prompt]",
        f"{'─'*60}",
        asst_sys,
        f"{'━'*60}",
    )

    user_history: list[dict] = [{"role": "system", "content": user_sys}]
    asst_history: list[dict] = [{"role": "system", "content": asst_sys}]
    conversation_lines: list[str] = []

    for turn in range(n_turns):
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


# ──────────────────────────────────────────────────────────────────────────────
# CSV schema
# ──────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "text", "label", "source", "primary_signal",
    "escalation_stage", "register", "language", "persona_id",
    "signals", "hardness_track",
]


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def generate(
    *,
    client: PrivateClient,
    judge_client: AnthropicClient,
    target: int,
    subtle_frac: float,
    persona_bank: list[dict],
    rng: random.Random,
    writer: "csv.DictWriter[str]",
    fh: "IO[str]",
    dry_run: bool = False,
    verbose: bool = False,
    existing_count: int = 0,
) -> int:
    remaining = max(0, target - existing_count)
    if remaining == 0:
        LOGGER.info("CAMEL-hard: already at %d/%d — skipping.", existing_count, target)
        return 0
    LOGGER.info("CAMEL-hard: need %d more (have %d/%d).", remaining, existing_count, target)

    n_written = 0
    n_failed  = 0
    i         = 0

    while n_written < remaining:
        # Assign track
        track = "subtle" if rng.random() < subtle_frac else "escalating"

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
            conversation = f"[DRY RUN CAMEL-HARD] track={track} signal={signal} stage={stage}"
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
                            track=track,
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
                        LOGGER.warning(
                            "CAMEL-hard dialogue error (attempt %d): %s", judge_attempt, exc
                        )
                        break

                    try:
                        judge_result = _call_camel_judge(
                            judge_client, conv_candidate, signal, track, n_turns
                        )
                        key_score = (
                            judge_result.get("subtlety")
                            if track == "subtle"
                            else judge_result.get("escalation_arc")
                        )
                        vprint(verbose,
                            f"[1b-hard JUDGE attempt {judge_attempt+1}  track={track}]",
                            f"  signal_presence={judge_result.get('signal_presence')}  "
                            f"{'subtlety' if track == 'subtle' else 'escalation_arc'}={key_score}  "
                            f"realism={judge_result.get('realism')}  "
                            f"accept={judge_result.get('accept')}",
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "CAMEL-hard judge error (attempt %d): %s — accepting anyway.", judge_attempt, exc
                        )
                        conversation = conv_candidate
                        break

                    if _camel_judge_accepts(judge_result, track):
                        conversation = conv_candidate
                        break
                    else:
                        user_feedback    = judge_result.get("signal_feedback", "")
                        support_feedback = judge_result.get("support_feedback", "")
                        LOGGER.info(
                            "  judge rejected (attempt %d/%d) track=%s  signal_p=%d  realism=%d  %s=%s",
                            judge_attempt + 1, _CAMEL_MAX_JUDGE_RETRIES + 1,
                            track,
                            judge_result.get("signal_presence", 0),
                            judge_result.get("realism", 0),
                            "subtlety" if track == "subtle" else "escalation_arc",
                            judge_result.get("subtlety" if track == "subtle" else "escalation_arc", 0),
                        )

                if conversation is None:
                    raise RuntimeError(
                        f"All {_CAMEL_MAX_JUDGE_RETRIES + 1} judge attempts failed."
                    )

                signal_dict = annotate_signals(client, conversation)
                signal_dict[signal] = 1

            except Exception as exc:
                n_failed += 1
                LOGGER.warning("CAMEL-hard conversation failed (row %d): %s", i, exc)
                time.sleep(2)
                if n_failed > remaining // 2:
                    LOGGER.error("Too many failures — stopping.")
                    break
                continue

        high_risk_any = int(any(v == 1 for v in signal_dict.values()))
        row = {
            "text":             conversation,
            "label":            high_risk_any,
            "source":           f"camel_hard_{track}",
            "primary_signal":   signal,
            "escalation_stage": stage,
            "register":         register,
            "language":         language,
            "persona_id":       persona.get("id", ""),
            "signals":          json.dumps(signal_dict),
            "hardness_track":   track,
        }
        append_row(writer, fh, row)
        n_written += 1
        i += 1
        LOGGER.info(
            "[CAMEL-hard %d/%d]  track=%-10s  signal=%-25s  lang=%s",
            n_written, remaining, track, signal, language,
        )

    return n_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1b-hard: CAMEL dual-agent generation with enforced subtlety/escalation."
    )
    parser.add_argument("--target",      type=int,   default=300,
                        help="Target conversation count")
    parser.add_argument("--subtle_frac", type=float, default=0.5,
                        help="Fraction of target to generate as Track A (subtle). "
                             "Remainder are Track B (escalating).")
    parser.add_argument("--output",      default="datasets/generated_camel_hard.csv",
                        help="Output CSV path")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dry_run",     action="store_true",
                        help="Simulate without API calls")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print system prompts and judge reasoning to stdout")
    parser.add_argument("--append",      action="store_true",
                        help="Append to existing output; resume from where we left off")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)

    existing_count = count_csv_rows(output_path) if args.append else 0
    if existing_count:
        LOGGER.info("Append mode: found %d existing rows in %s.", existing_count, output_path)

    if not PERSONA_BANK_PATH.exists():
        LOGGER.error("Persona bank not found: %s", PERSONA_BANK_PATH)
        return 1
    import json as _json
    with open(PERSONA_BANK_PATH, "r", encoding="utf-8") as f:
        persona_bank: list[dict] = _json.load(f)
    LOGGER.info("Loaded %d personas.", len(persona_bank))

    client       = PrivateClient()
    judge_client = AnthropicClient()
    fh, writer   = init_csv(output_path, FIELDNAMES, append=args.append)
    try:
        n = generate(
            client=client,
            judge_client=judge_client,
            target=args.target,
            subtle_frac=args.subtle_frac,
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

    LOGGER.info("✓ Wrote %d rows this run to %s", n, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
