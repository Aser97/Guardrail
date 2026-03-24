"""
scripts/generate_pair_hard.py
Phase 2g-hard — PAIR adversarial generation with enforced hardness constraints.

Extends generate_pair.py with stricter judge gates and redesigned criteria for
both hard positives and hard negatives.

HARD POSITIVE — two tracks (mirrors generate_camel_hard.py):

  Track A — SUBTLE:
    Signal present but expressed entirely through implication, metaphor, or
    topic adjacency. Never named explicitly.
    Judge gate: signal_presence ≥ 7 AND subtlety ≥ 7 AND realism ≥ 7

  Track B — ESCALATING:
    Conversation arc starts low-risk and gradually builds to signal peak.
    First third of turns must be indistinguishable from a benign conversation.
    Judge gate: signal_presence ≥ 7 AND escalation_arc ≥ 7 AND realism ≥ 7

HARD NEGATIVE — redesigned criteria:

  mimicry ≥ 7: The conversation must genuinely look high-risk at surface level
    (same vocabulary, topic, or emotional register as real crisis conversations).
    Low-mimicry negatives are easy for a classifier and not worth generating.

  absence_clarity ≥ 8 (strict threshold): A careful reader must be completely
    certain no real signal is present. Threshold is higher than mimicry by design —
    better to reject a good-looking negative than to accidentally poison the
    negative class with a real crisis.

  realism ≥ 7: Sounds like a real youth, not AI-generated.

Both positive tracks and the negative track use chain-of-thought judge prompts.

Output: datasets/generated_pair_hard.csv
  Includes `hardness_track` column: "subtle", "escalating", or "adversarial_negative"

Usage:
    python project/scripts/generate_pair_hard.py \\
        --target 200 \\
        --max_iter 5 \\
        --output datasets/generated_pair_hard.csv \\
        [--subtle_frac 0.4] [--escalating_frac 0.4]  \\
        --seed 42 [--append]
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
import time
from pathlib import Path
from typing import Optional

from config import (
    SIGNALS,
    SIGNAL_DESCRIPTIONS,
    PHRASE_CATALOG,
    TEMPERATURE,
    LANGUAGE_WEIGHTS,
)
from typing import IO
from llm_client import PrivateClient, AnthropicClient, CLAUDE_SONNET, PRIVATE_MISTRAL
from utils import init_csv, append_row, count_csv_rows_by_field, vprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

MISTRAL = PRIVATE_MISTRAL

# ──────────────────────────────────────────────────────────────────────────────
# Generator prompts (Mistral)
# ──────────────────────────────────────────────────────────────────────────────

_GENERATOR_SUBTLE = """\
You are generating a HARD POSITIVE training conversation for a youth mental health safety classifier.
The conversation must make the target signal present but NEVER explicitly stated.

Signal to embed: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}
Key phrase (draw from naturally — never quote verbatim): {KEY_PHRASE}
Taxonomy category: {CATEGORY}
Language: {LANGUAGE}
Turn count: {N_TURNS}

SUBTLETY REQUIREMENT — the most important rule:
  The user must NEVER name the signal, use clinical terminology, or state it directly.
  Embed the signal exclusively through:
    • Oblique language: "I don't know why I even try anymore"
    • Metaphor or imagery: "it's like there's nothing left to hold onto"
    • Topic adjacency: circling the real issue without ever landing on it
    • Third-person deflection: "just asking for a friend who feels that way sometimes"
    • Minimization: "it's stupid, forget it, I shouldn't have said anything"
  A trained safety expert must be able to detect it. A keyword filter must not.

User register: youth aged 13-22, informal, slang-heavy, fragmented sentences.
Assistant: professional support chatbot — warm, empathetic, boundaried. No emojis,
no exclamation chains, no terms of endearment. 2–3 sentences max.

{FEEDBACK_BLOCK}

Output: full conversation only (user: / assistant: format). No explanation.
"""

_GENERATOR_ESCALATING = """\
You are generating a HARD POSITIVE training conversation for a youth mental health safety classifier.
The conversation must follow a SLOW ESCALATION ARC — starting low-risk and building to crisis.

Signal to embed: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}
Key phrase (draw from naturally — never quote verbatim): {KEY_PHRASE}
Taxonomy category: {CATEGORY}
Language: {LANGUAGE}
Turn count: {N_TURNS}

ESCALATION ARC REQUIREMENT — the most important rule:
  Divide the conversation into three phases:

  Phase 1 — ENTRY (turns 1–{EARLY}):
    The user discusses something ordinary and surface-level. A reader would not
    suspect anything unusual. No visible risk signal.
    Examples: complaining about school, a mild argument with a friend, being tired.

  Phase 2 — BUILD (turns {EARLY_P1}–{EARLY_MIDDLE}):
    The emotional intensity increases gradually. Vague hints emerge — something deeper
    beneath the surface — but nothing explicit. Easy to dismiss as normal stress.

  Phase 3 — PEAK (turns {EARLY_MIDDLE_P1}–{N_TURNS}):
    The signal "{SIGNAL_NAME}" becomes detectable, expressed with some indirection.
    The distress is real and no longer fully hidden. The arc completes here.

  The arc must feel natural and psychologically real, not mechanical or sudden.

User register: youth aged 13-22, informal, slang-heavy, fragmented sentences.
Assistant: professional support chatbot — warm, empathetic, boundaried. No emojis,
no exclamation chains, no terms of endearment. 2–3 sentences max.

{FEEDBACK_BLOCK}

Output: full conversation only (user: / assistant: format). No explanation.
"""

_GENERATOR_NEGATIVE = """\
You are generating a HARD NEGATIVE training conversation for a youth mental health safety classifier.
This conversation must LOOK high-risk on the surface but be GENUINELY LOW-RISK on careful reading.

Target signal it should SUPERFICIALLY resemble: {SIGNAL_NAME}
Category: {CATEGORY}
Language: {LANGUAGE}
Turn count: {N_TURNS}

MIMICRY REQUIREMENT — the most important rule:
  Use the vocabulary, topics, and emotional register of "{SIGNAL_NAME}" conversations,
  but ensure the underlying meaning is definitively NOT a real crisis.

  The best hard negatives use one of these archetypes:
    • Hyperbole/dark humor: "I literally want to die, this exam is killing me"
      → uses finality_language vocabulary but is obviously hyperbolic
    • Normative adolescent distress: genuine sadness or frustration within a normal
      developmental range — real emotion, not clinical risk
    • Third-party/academic framing: "I was reading about [topic] and wondering why
      people feel that way" — topic overlap, no personal identification
    • Fiction or creative writing context: "my character in this story is going through..."
    • Resolved or past-tense distress: "I was really struggling last week but I
      actually feel fine now"
    • Frustrated venting without hopelessness: "I hate everything about today" said
      about a failed test or minor conflict

  The conversation must be DEFINITIVELY LOW-RISK. No ambiguity. A careful reader
  must be completely certain no real signal is present.

User register: youth aged 13-22, informal, slang-heavy, fragmented sentences.
Assistant: professional support chatbot — warm, empathetic, boundaried. No emojis,
no exclamation chains, no terms of endearment. 2–3 sentences max.

{FEEDBACK_BLOCK}

Output: full conversation only (user: / assistant: format). No explanation.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Judge prompts — chain-of-thought, track-specific acceptance gates
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_SUBTLE = """\
You are an expert safety data evaluator for a youth mental health classifier.
Your task: assess whether this is a high-quality SUBTLE hard positive example.

Target signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}

Conversation:
{CONVERSATION_TEXT}

Think step by step before scoring.

STEP 1 — Locate signal evidence:
  Read each user turn carefully.
  Which turns, if any, contain evidence of "{SIGNAL_NAME}"? Quote the exact phrases.
  Is the signal genuinely detectable by a trained safety expert, or is it absent?

STEP 2 — Assess subtlety:
  For each identified turn from Step 1:
  → Direct/explicit (score 1–3): signal is named or stated outright
  → Indirect but recognizable (score 4–6): euphemistic, but a lay reader might catch it
  → Highly indirect (score 7–10): requires expert inference; metaphor, implication, or
    circumlocution; a keyword filter would miss it entirely
  What is the overall subtlety level across the conversation?

STEP 3 — Assess realism:
  Does the user sound like a real youth aged 13–22?
  Does the assistant maintain professional, boundaried tone (no emojis, no peer-like familiarity)?

STEP 4 — Score and decide:
  signal_presence (1–10): Detectable by a safety expert? → REQUIRED ≥ 7
  subtlety (1–10): How indirect? → REQUIRED ≥ 7 (HARD GATE — any explicit naming = reject)
  realism (1–10): Both sides realistic? → REQUIRED ≥ 7
  accept: true only if ALL THREE thresholds are met.

If rejected, provide ONE sentence of concrete actionable feedback.

Write your step-by-step reasoning first, then end with JSON on a new line:
{{"signal_presence": X, "subtlety": X, "realism": X, "accept": true/false, "feedback": "..."}}
"""

_JUDGE_ESCALATING = """\
You are an expert safety data evaluator for a youth mental health classifier.
Your task: assess whether this is a high-quality ESCALATING hard positive example.

Target signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}
Total turns in conversation: {N_TURNS}

Conversation:
{CONVERSATION_TEXT}

Think step by step before scoring.

STEP 1 — Map the arc:
  Identify approximately where each phase begins and ends.
  Describe the emotional tone and risk level in each third of the conversation:
    Early ({EARLY} turns): What is the user talking about? Any visible risk?
    Middle ({MIDDLE} turns): How does it evolve? Any hints?
    Late ({LATE} turns): Where does the signal peak? Is it clearly detectable?

STEP 2 — Evaluate arc quality:
  Early phase: Could this opening come from a completely benign, low-risk conversation?
  Transitions: Does each phase flow naturally into the next, or are shifts sudden?
  Peak: Is the signal "{SIGNAL_NAME}" clearly present in the final phase?

STEP 3 — Assess signal and realism:
  Is the signal genuinely detectable in the late portion?
  Does the user sound like a real youth throughout all three phases?
  Does the assistant maintain appropriate professional tone?

STEP 4 — Score and decide:
  signal_presence (1–10): Detectable in the conversation? → REQUIRED ≥ 7
  escalation_arc (1–10): Quality of gradual trajectory.
    10 = early turns are completely benign, arc is smooth, peak is clear.
    1  = distress obvious from turn 1, no real arc.
    → REQUIRED ≥ 7
  realism (1–10): Both sides realistic throughout? → REQUIRED ≥ 7
  accept: true only if ALL THREE thresholds are met.

If rejected, provide ONE sentence of concrete actionable feedback.

Write your step-by-step reasoning first, then end with JSON on a new line:
{{"signal_presence": X, "escalation_arc": X, "realism": X, "accept": true/false, "feedback": "..."}}
"""

_JUDGE_NEGATIVE = """\
You are an expert safety data evaluator for a youth mental health classifier.
Your task: assess whether this is a high-quality HARD NEGATIVE example.

A hard negative must (1) look high-risk at surface level, and (2) be definitively
low-risk on careful reading. Both properties must hold simultaneously.

Target signal it should superficially resemble: {SIGNAL_NAME}

Conversation:
{CONVERSATION_TEXT}

Think step by step before scoring.

STEP 1 — Identify surface mimicry:
  What vocabulary, topics, or emotional register from "{SIGNAL_NAME}" appears in this conversation?
  Quote specific phrases that could mislead a superficial classifier.
  Would a naive keyword-based classifier or a distracted human likely flag this?

STEP 2 — Verify genuine low-risk status:
  Read carefully for any real signal of genuine crisis or distress.
  Is the user's distress definitively resolved, fictional, academic, or hyperbolic?
  Is there ANY ambiguity about whether a real crisis is present?
  (Even slight ambiguity means this should be rejected — we need certainty here.)

STEP 3 — Assess realism:
  Does this sound like a real youth aged 13–22 in a genuine conversation?

STEP 4 — Score and decide:
  mimicry (1–10): How convincingly does this resemble a real "{SIGNAL_NAME}" conversation at surface level?
    1 = obviously low-risk on first read. 10 = would fool a distracted human reviewer.
    → REQUIRED ≥ 7 (low-mimicry negatives are not useful training data)
  absence_clarity (1–10): How unambiguously is the signal ABSENT?
    1 = real signal clearly present (reject). 10 = definitively low-risk with no ambiguity.
    → REQUIRED ≥ 8 (strict — we cannot risk labeling a real crisis as negative)
  realism (1–10): Both sides sound realistic? → REQUIRED ≥ 7
  accept: true only if ALL THREE thresholds are met.

If rejected, provide ONE sentence of concrete actionable feedback.

Write your step-by-step reasoning first, then end with JSON on a new line:
{{"mimicry": X, "absence_clarity": X, "realism": X, "accept": true/false, "feedback": "..."}}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Generator call
# ──────────────────────────────────────────────────────────────────────────────

def _call_generator(
    client: PrivateClient,
    entry: dict,
    track: str,
    language: str,
    n_turns: int,
    feedback_history: list[str],
) -> str:
    feedback_block = ""
    if feedback_history:
        feedback_block = "Prior attempt feedback:\n" + "\n".join(
            f"  - {fb}" for fb in feedback_history[-3:]
        )

    if track == "subtle":
        prompt = _GENERATOR_SUBTLE.format(
            SIGNAL_NAME=entry["signal"],
            SIGNAL_DEFINITION=SIGNAL_DESCRIPTIONS.get(entry["signal"], ""),
            KEY_PHRASE=entry["phrase"],
            CATEGORY=entry["category"],
            LANGUAGE=language,
            N_TURNS=n_turns,
            FEEDBACK_BLOCK=feedback_block,
        )
    elif track == "escalating":
        early  = max(4, n_turns // 3)
        middle = max(4, n_turns // 3)
        prompt = _GENERATOR_ESCALATING.format(
            SIGNAL_NAME=entry["signal"],
            SIGNAL_DEFINITION=SIGNAL_DESCRIPTIONS.get(entry["signal"], ""),
            KEY_PHRASE=entry["phrase"],
            CATEGORY=entry["category"],
            LANGUAGE=language,
            N_TURNS=n_turns,
            EARLY=early,
            EARLY_P1=early + 1,
            EARLY_MIDDLE=early + middle,
            EARLY_MIDDLE_P1=early + middle + 1,
            FEEDBACK_BLOCK=feedback_block,
        )
    else:  # adversarial_negative
        prompt = _GENERATOR_NEGATIVE.format(
            SIGNAL_NAME=entry["signal"],
            CATEGORY=entry["category"],
            LANGUAGE=language,
            N_TURNS=n_turns,
            FEEDBACK_BLOCK=feedback_block,
        )

    return client.complete(
        model=MISTRAL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE["user_turn"],
        max_tokens=3000,
    ).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Judge call
# ──────────────────────────────────────────────────────────────────────────────

def _call_judge(
    anthropic_client: AnthropicClient,
    conversation: str,
    signal: str,
    track: str,
    n_turns: int,
) -> dict:
    defn = SIGNAL_DESCRIPTIONS.get(signal, "")
    if track == "subtle":
        prompt = _JUDGE_SUBTLE.format(
            SIGNAL_NAME=signal,
            SIGNAL_DEFINITION=defn,
            CONVERSATION_TEXT=conversation,
        )
    elif track == "escalating":
        early  = max(4, n_turns // 3)
        middle = max(4, n_turns // 3)
        late   = n_turns - early - middle
        prompt = _JUDGE_ESCALATING.format(
            SIGNAL_NAME=signal,
            SIGNAL_DEFINITION=defn,
            CONVERSATION_TEXT=conversation,
            N_TURNS=n_turns,
            EARLY=early,
            MIDDLE=middle,
            LATE=late,
        )
    else:  # adversarial_negative
        prompt = _JUDGE_NEGATIVE.format(
            SIGNAL_NAME=signal,
            CONVERSATION_TEXT=conversation,
        )

    raw = anthropic_client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE["annotation"],
        max_tokens=800,   # extra room for CoT reasoning
    ).strip()
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"Judge returned unparseable response: {raw[:300]}")
    return json.loads(m.group())


def _judge_accepts(result: dict, track: str) -> bool:
    if not result.get("accept", False):
        return False
    if track == "subtle":
        return (
            result.get("signal_presence", 0) >= 7 and
            result.get("subtlety", 0)         >= 7 and
            result.get("realism", 0)           >= 7
        )
    elif track == "escalating":
        return (
            result.get("signal_presence", 0)  >= 7 and
            result.get("escalation_arc", 0)   >= 7 and
            result.get("realism", 0)           >= 7
        )
    else:  # adversarial_negative
        return (
            result.get("mimicry", 0)           >= 7 and
            result.get("absence_clarity", 0)   >= 8 and   # stricter threshold
            result.get("realism", 0)           >= 7
        )


# ──────────────────────────────────────────────────────────────────────────────
# PAIR loop for a single conversation
# ──────────────────────────────────────────────────────────────────────────────

def _pair_loop(
    entry: dict,
    track: str,
    language: str,
    n_turns: int,
    max_iter: int,
    generator_client: PrivateClient,
    judge_client: AnthropicClient,
    rng: random.Random,
    verbose: bool = False,
) -> Optional[dict]:
    """
    Run the PAIR loop for one (entry, track) pair.
    Returns a master-schema row on success, None on failure.
    """
    feedback_history: list[str] = []
    signal = entry["signal"]

    for attempt in range(1, max_iter + 1):
        LOGGER.info("  attempt %d/%d  signal=%s  track=%s", attempt, max_iter, signal, track)
        try:
            conversation = _call_generator(
                generator_client, entry, track, language, n_turns, feedback_history
            )
            vprint(verbose,
                f"\n{'━'*60}",
                f"[2g-hard GENERATOR  attempt={attempt}  track={track}  "
                f"signal={signal}  lang={language}]",
                f"{'─'*60}",
                conversation,
                f"{'━'*60}",
            )
        except Exception as exc:
            LOGGER.warning("  generator error: %s", exc)
            continue

        try:
            judge_result = _call_judge(judge_client, conversation, signal, track, n_turns)
            vprint(verbose,
                f"[2g-hard JUDGE result  track={track}]",
                f"  {judge_result}",
            )
        except Exception as exc:
            LOGGER.warning("  judge error: %s", exc)
            continue

        accepted = _judge_accepts(judge_result, track)
        feedback = judge_result.get("feedback", "")
        if feedback:
            feedback_history.append(feedback)

        # Log relevant scores per track
        if track == "subtle":
            scores = {
                "signal_presence": judge_result.get("signal_presence"),
                "subtlety":        judge_result.get("subtlety"),
                "realism":         judge_result.get("realism"),
            }
        elif track == "escalating":
            scores = {
                "signal_presence": judge_result.get("signal_presence"),
                "escalation_arc":  judge_result.get("escalation_arc"),
                "realism":         judge_result.get("realism"),
            }
        else:
            scores = {
                "mimicry":          judge_result.get("mimicry"),
                "absence_clarity":  judge_result.get("absence_clarity"),
                "realism":          judge_result.get("realism"),
            }
        LOGGER.info("  judge: accept=%s  scores=%s", accepted, scores)

        if accepted:
            is_positive = track in ("subtle", "escalating")
            signals_dict = {s: 0 for s in SIGNALS}
            label = 1
            if is_positive:
                signals_dict[signal] = 1
            else:
                label = 0

            row = {
                "text":             conversation,
                "label":            label,
                "source":           f"pair_hard_{track}",
                "primary_signal":   signal if is_positive else "none",
                "escalation_stage": "",
                "register":         "",
                "language":         language,
                "persona_id":       "",
                "signals":          json.dumps(signals_dict),
                "category":         entry.get("category", ""),
                "hardness_track":   track,
            }
            vprint(verbose,
                f"[2g-hard ACCEPTED ✓]  track={track}  label={label}  signal={signal}\n{'━'*60}\n"
            )
            return row

        LOGGER.info("  rejected — will retry with feedback.")
        vprint(verbose, f"[2g-hard REJECTED ✗]  feedback: {feedback}")

    LOGGER.info("  FAILED after %d attempts.", max_iter)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 2g-hard: PAIR adversarial generation with enforced hardness constraints."
    )
    p.add_argument("--target",          type=int,   default=200,
                   help="Total rows to generate across all tracks")
    p.add_argument("--max_iter",        type=int,   default=5,
                   help="Max PAIR iterations per conversation")
    p.add_argument("--subtle_frac",     type=float, default=0.40,
                   help="Fraction of target: Track A (subtle hard positives)")
    p.add_argument("--escalating_frac", type=float, default=0.40,
                   help="Fraction of target: Track B (escalating hard positives). "
                        "Remainder = adversarial negatives.")
    p.add_argument("--output",          default="datasets/generated_pair_hard.csv")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--verbose",         action="store_true",
                   help="Print generator prompts, conversations, and judge reasoning.")
    p.add_argument("--append",          action="store_true",
                   help="Append to existing output; resume from already-written rows.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    generator_client = PrivateClient()
    judge_client     = AnthropicClient()
    output_path = Path(args.output)

    # Resume: count existing rows by track
    existing_by_source = count_csv_rows_by_field(output_path, "source") if args.append else {}
    existing_subtle    = existing_by_source.get("pair_hard_subtle",     0)
    existing_escalat   = existing_by_source.get("pair_hard_escalating", 0)
    existing_neg       = existing_by_source.get("pair_hard_adversarial_negative", 0)

    n_subtle    = max(0, int(args.target * args.subtle_frac)     - existing_subtle)
    n_escalat   = max(0, int(args.target * args.escalating_frac) - existing_escalat)
    neg_frac    = max(0.0, 1.0 - args.subtle_frac - args.escalating_frac)
    n_negative  = max(0, int(args.target * neg_frac)             - existing_neg)

    LOGGER.info(
        "PAIR-hard targets: %d subtle + %d escalating + %d adversarial_negative "
        "(resume: had %d+%d+%d)",
        n_subtle, n_escalat, n_negative,
        existing_subtle, existing_escalat, existing_neg,
    )

    langs   = list(LANGUAGE_WEIGHTS.keys())
    weights = list(LANGUAGE_WEIGHTS.values())

    FIELDNAMES = [
        "text", "label", "source", "primary_signal",
        "escalation_stage", "register", "language", "persona_id",
        "signals", "category", "hardness_track",
    ]
    fh, writer = init_csv(output_path, FIELDNAMES, append=args.append)

    attempts_log: list[dict] = []
    total_written = 0

    def _run_batch(track: str, n: int) -> None:
        nonlocal total_written
        generated = 0
        while generated < n:
            entry    = rng.choice(PHRASE_CATALOG)
            language = rng.choices(langs, weights=weights, k=1)[0]
            n_turns  = rng.randint(16, 20)

            t0 = time.time()
            row = _pair_loop(
                entry, track, language, n_turns, args.max_iter,
                generator_client, judge_client, rng, verbose=args.verbose,
            )
            elapsed = time.time() - t0

            attempts_log.append({
                "track": track, "signal": entry["signal"],
                "accepted": row is not None, "elapsed_s": round(elapsed, 1),
            })

            if row:
                append_row(writer, fh, row)
                generated     += 1
                total_written += 1
                LOGGER.info(
                    "[%s] %d/%d  signal=%s  lang=%s  (%.0fs)",
                    track, generated, n, entry["signal"], language, elapsed,
                )

    try:
        LOGGER.info("=== Phase 2g-hard: subtle hard positives (need %d) ===", n_subtle)
        _run_batch("subtle", n_subtle)

        LOGGER.info("=== Phase 2g-hard: escalating hard positives (need %d) ===", n_escalat)
        _run_batch("escalating", n_escalat)

        LOGGER.info("=== Phase 2g-hard: adversarial negatives (need %d) ===", n_negative)
        _run_batch("adversarial_negative", n_negative)
    finally:
        fh.close()

    LOGGER.info("✓ Wrote %d rows this run to %s", total_written, output_path)

    log_path = output_path.with_suffix(".attempts.json")
    existing_log: list[dict] = []
    if args.append and log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                existing_log = json.load(lf)
        except Exception:
            pass
    with open(log_path, "w", encoding="utf-8") as lf:
        json.dump(existing_log + attempts_log, lf, indent=2)
    LOGGER.info("Attempt log → %s (%d total entries)", log_path, len(existing_log) + len(attempts_log))

    return 0


if __name__ == "__main__":
    sys.exit(main())
