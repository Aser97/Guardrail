"""
scripts/generate_pair.py
Phase 2g — PAIR-inspired adversarial generation.

Generator : Mistral Large 3   (hackathon endpoint, free)
Judge     : Claude 3.7 Sonnet (Anthropic API, ~$20-30 total)

Two modes:
  hard_positive      — signal is present but expressed INDIRECTLY (subtle)
  adversarial_negative — mimics high-risk surface language but is genuinely low-risk

Loop per row:
  1. Generator produces a conversation embedding the target signal subtly.
  2. Judge scores on three axes (1-10):
       signal_presence ≥ 7 AND realism ≥ 7 → ACCEPT
       subtlety is tracked but NOT a pass/fail gate (we want a mix)
  3. If rejected: judge's feedback is added to the prompt history; retry up to max_iter.

Sampling unit: PHRASE_CATALOG entry  →  phrase + signal + category (same as generation scripts).

Output columns match internal master CSV schema.

Usage:
    python project/scripts/generate_pair.py \\
        --target   200 \\
        --max_iter 5 \\
        --output   datasets/generated_pair.csv \\
        --seed     42
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
import time
from pathlib import Path
from typing import Optional


from config import (
    SIGNALS,
    SIGNAL_DESCRIPTIONS,
    PHRASE_CATALOG,
    TEMPERATURE,
    TAXONOMY_CATEGORIES,
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

MISTRAL = PRIVATE_MISTRAL  # Mistral Large via Mistral AI API (private key)

# ──────────────────────────────────────────────────────────────────────────────
# Generator prompts (Mistral)
# ──────────────────────────────────────────────────────────────────────────────

_GENERATOR_HARD_POSITIVE = """\
You are generating a SUBTLE high-risk training conversation for a youth mental health \
safety classifier.

Signal to embed: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}
Key phrase (express naturally — do NOT quote verbatim): {KEY_PHRASE}

Taxonomy category: {CATEGORY}
Language: {LANGUAGE}
Turn count: {N_TURNS}

SUBTLETY REQUIREMENT: The signal must be detectable by a trained safety expert but NOT \
by simple keyword matching. Use metaphor, indirect phrasing, euphemism, topic drift, \
or third-person framing to embed the signal. The user should never name the signal directly.

User register: youth aged 13-22, informal, slang-heavy, fragmented sentences.
Assistant is a professional support chatbot: warm and empathetic, but boundaried. No emojis, no exclamation chains, no terms of endearment. Responses are 2–3 sentences maximum. The assistant cannot resolve the crisis.

{FEEDBACK_BLOCK}

Output: full conversation only (user: / assistant: format). No explanation.
"""

_GENERATOR_HARD_NEGATIVE = """\
You are generating an ADVERSARIAL NEGATIVE training conversation for a youth mental health \
safety classifier.

This conversation should SUPERFICIALLY resemble high-risk language related to: {SIGNAL_NAME}
But it must be GENUINELY LOW-RISK — the signal is NOT actually present.

Examples of adversarial negatives:
  - "I could literally die" used as hyperbole for excitement
  - Discussing a dark fiction story or video game
  - Joking dark humour with friends ("kms lol")
  - Academic frustration that sounds hopeless but isn't
  - Genuine curiosity about the chatbot's safety features

Category: {CATEGORY}
Language: {LANGUAGE}
Turn count: {N_TURNS}
Assistant register: professional support chatbot — warm but boundaried, no emojis, no terms of endearment, 2–3 sentence responses.

{FEEDBACK_BLOCK}

Output: full conversation only (user: / assistant: format). No explanation.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Judge prompt (Claude 3.7 Sonnet)
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_HARD_POSITIVE = """\
Evaluate this youth chatbot conversation for safety training data quality.
Target signal: {SIGNAL_NAME}
Definition: {SIGNAL_DEFINITION}

Conversation:
{CONVERSATION_TEXT}

Score on three axes from 1 to 10:

1. signal_presence: Is {SIGNAL_NAME} genuinely detectable by a trained safety expert?
   (1 = not present at all, 10 = unambiguously present)
   → REQUIRED: must score ≥ 7 to accept

2. subtlety: How indirect / non-obvious is the signal expression?
   (1 = very explicit and direct, 10 = very subtle, requires inference)
   → INFORMATIONAL ONLY — we want a mix (not all subtle)

3. realism: Does this sound like a real youth aged 13-22 in distress, not an AI-generated example?
   (1 = clearly synthetic / unnatural, 10 = highly realistic voice and register)
   → REQUIRED: must score ≥ 7 to accept

If rejected: provide ONE sentence of concrete, actionable feedback to improve the next attempt.

Output JSON only:
{{"signal_presence": X, "subtlety": X, "realism": X, "accept": true/false, "feedback": "..."}}
"""

_JUDGE_HARD_NEGATIVE = """\
Evaluate this youth chatbot conversation for adversarial negative quality.
It should APPEAR to contain {SIGNAL_NAME}-related language but be GENUINELY LOW-RISK.

Conversation:
{CONVERSATION_TEXT}

Score on three axes from 1 to 10:

1. negative_validity: Is this conversation genuinely low-risk despite surface-level resemblance?
   (1 = clearly high-risk, 10 = genuinely safe)
   → REQUIRED: must score ≥ 7 to accept

2. surface_similarity: How much does it superficially resemble a {SIGNAL_NAME} conversation?
   (1 = no resemblance, 10 = very convincing surface mimicry)
   → REQUIRED: must score ≥ 6 to be useful as a hard negative

3. realism: Does this sound like a real youth aged 13-22, not AI-generated?
   (1 = synthetic, 10 = realistic)
   → REQUIRED: must score ≥ 7 to accept

If rejected: provide ONE sentence of actionable feedback.

Output JSON only:
{{"negative_validity": X, "surface_similarity": X, "realism": X, "accept": true/false, "feedback": "..."}}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Judge call
# ──────────────────────────────────────────────────────────────────────────────

def _call_judge(
    anthropic_client: AnthropicClient,
    conversation: str,
    signal: str,
    mode: str,
) -> dict:
    """
    Call Claude 3.7 Sonnet as the PAIR judge.
    Returns parsed JSON dict with accept, scores, and feedback.
    """
    definition = SIGNAL_DESCRIPTIONS.get(signal, "")
    if mode == "hard_positive":
        prompt = _JUDGE_HARD_POSITIVE.format(
            SIGNAL_NAME=signal,
            SIGNAL_DEFINITION=definition,
            CONVERSATION_TEXT=conversation,
        )
    else:
        prompt = _JUDGE_HARD_NEGATIVE.format(
            SIGNAL_NAME=signal,
            CONVERSATION_TEXT=conversation,
        )

    raw = anthropic_client.complete(
        model=CLAUDE_SONNET,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE["judge"],
        max_tokens=300,
    ).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"Judge returned unparseable response: {raw[:200]}")
    return json.loads(m.group())


def _judge_accepts(result: dict, mode: str) -> bool:
    """Determine whether the judge accepts this conversation."""
    if mode == "hard_positive":
        return (
            result.get("accept", False) and
            result.get("signal_presence", 0) >= 7 and
            result.get("realism", 0) >= 7
        )
    else:  # adversarial_negative
        return (
            result.get("accept", False) and
            result.get("negative_validity", 0) >= 7 and
            result.get("surface_similarity", 0) >= 6 and
            result.get("realism", 0) >= 7
        )


# ──────────────────────────────────────────────────────────────────────────────
# Generator call
# ──────────────────────────────────────────────────────────────────────────────

def _call_generator(
    client: PrivateClient,
    entry: dict,
    mode: str,
    language: str,
    n_turns: int,
    feedback_history: list[str],
    rng: random.Random,
) -> str:
    """Call Mistral to generate a conversation. Returns raw conversation text."""
    feedback_block = ""
    if feedback_history:
        feedback_block = "Prior attempt feedback:\n" + "\n".join(
            f"  - {fb}" for fb in feedback_history[-3:]  # keep last 3
        )

    if mode == "hard_positive":
        prompt = _GENERATOR_HARD_POSITIVE.format(
            SIGNAL_NAME=entry["signal"],
            SIGNAL_DEFINITION=SIGNAL_DESCRIPTIONS.get(entry["signal"], ""),
            KEY_PHRASE=entry["phrase"],
            CATEGORY=entry["category"],
            LANGUAGE=language,
            N_TURNS=n_turns,
            FEEDBACK_BLOCK=feedback_block,
        )
    else:
        prompt = _GENERATOR_HARD_NEGATIVE.format(
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
# PAIR loop for a single target
# ──────────────────────────────────────────────────────────────────────────────

def _pair_loop(
    entry: dict,
    mode: str,
    language: str,
    n_turns: int,
    max_iter: int,
    generator_client: PrivateClient,
    judge_client: AnthropicClient,
    rng: random.Random,
    verbose: bool = False,
) -> Optional[dict]:
    """
    Run the PAIR loop for one (entry, mode) pair.
    Returns a master-schema row on success, None on failure after max_iter.
    """
    feedback_history: list[str] = []
    signal = entry["signal"]

    for attempt in range(1, max_iter + 1):
        LOGGER.info("  attempt %d/%d  signal=%s  mode=%s", attempt, max_iter, signal, mode)
        try:
            # --- Generator ---
            conversation = _call_generator(
                generator_client, entry, mode, language, n_turns, feedback_history, rng
            )
            vprint(verbose,
                f"\n{'━'*60}",
                f"[2g GENERATOR  attempt={attempt}  mode={mode}  signal={signal}  lang={language}]",
                f"{'─'*60}",
                conversation,
                f"{'━'*60}",
            )
        except Exception as exc:
            LOGGER.warning("  generator error: %s", exc)
            continue

        try:
            # --- Judge ---
            judge_result = _call_judge(judge_client, conversation, signal, mode)
            vprint(verbose,
                f"[2g JUDGE result]",
                f"  {judge_result}",
            )
        except Exception as exc:
            LOGGER.warning("  judge error: %s", exc)
            continue

        accepted = _judge_accepts(judge_result, mode)
        feedback = judge_result.get("feedback", "")
        if feedback:
            feedback_history.append(feedback)

        LOGGER.info(
            "  judge: accept=%s  scores=%s",
            accepted,
            {k: v for k, v in judge_result.items() if k != "feedback" and k != "accept"},
        )

        if accepted:
            signals_dict = {s: 0 for s in SIGNALS}
            label = 1
            if mode == "hard_positive":
                signals_dict[signal] = 1
            else:
                label = 0

            row = {
                "text":             conversation,
                "label":            label,
                "source":           f"pair_{mode}",
                "primary_signal":   signal if mode == "hard_positive" else "none",
                "escalation_stage": "",
                "register":         "",
                "language":         language,
                "persona_id":       "",
                "signals":          json.dumps(signals_dict),
                "category":         entry.get("category", ""),
            }
            vprint(verbose, f"[2g ACCEPTED ✓]  label={label}  signal={signal}\n{'━'*60}\n")
            return row

        LOGGER.info("  rejected — will retry with feedback.")
        vprint(verbose, f"[2g REJECTED ✗]  feedback: {feedback}")

    LOGGER.info("  FAILED after %d attempts.", max_iter)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 2g: PAIR adversarial generation (Mistral generator + Claude 3.7 Sonnet judge)."
    )
    p.add_argument("--target",      type=int, default=200,
                   help="Total rows to generate (hard positives + adversarial negatives)")
    p.add_argument("--max_iter",    type=int, default=5,
                   help="Max PAIR iterations per row")
    p.add_argument("--hard_positive_frac", type=float, default=0.70,
                   help="Fraction of target that are hard positives (rest = adversarial negatives)")
    p.add_argument("--output",      default="datasets/generated_pair.csv")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--verbose",     action="store_true",
                   help="Print generator prompt/response and judge verdict to stdout.")
    p.add_argument("--append",      action="store_true",
                   help="Append to existing output; reduce targets by already-written rows.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    generator_client = PrivateClient()   # Mistral Large via Mistral AI API
    judge_client     = AnthropicClient()  # Claude 3.7 Sonnet via Anthropic API
    output_path = Path(args.output)

    # Resume: count existing by mode
    existing_by_source = count_csv_rows_by_field(output_path, "source") if args.append else {}
    existing_hp = existing_by_source.get("pair_hard_positive", 0)
    existing_an = existing_by_source.get("pair_adversarial_negative", 0)

    n_hard_positive = max(0, int(args.target * args.hard_positive_frac) - existing_hp)
    n_hard_negative = max(0, (args.target - int(args.target * args.hard_positive_frac)) - existing_an)

    LOGGER.info(
        "PAIR targets: %d hard positives + %d adversarial negatives (resume: had %d+%d)",
        n_hard_positive, n_hard_negative, existing_hp, existing_an,
    )

    langs   = list(LANGUAGE_WEIGHTS.keys())
    weights = list(LANGUAGE_WEIGHTS.values())

    FIELDNAMES_PAIR = [
        "text", "label", "source", "primary_signal",
        "escalation_stage", "register", "language", "persona_id", "signals", "category",
    ]
    fh, writer = init_csv(output_path, FIELDNAMES_PAIR, append=args.append)

    attempts_log: list[dict] = []
    total_written = 0

    def _run_batch(mode: str, n: int) -> None:
        nonlocal total_written
        generated = 0
        while generated < n:
            entry    = rng.choice(PHRASE_CATALOG)
            language = rng.choices(langs, weights=weights, k=1)[0]
            n_turns  = rng.randint(16, 20)

            t0 = time.time()
            row = _pair_loop(
                entry, mode, language, n_turns, args.max_iter,
                generator_client, judge_client, rng, verbose=args.verbose,
            )
            elapsed = time.time() - t0

            attempts_log.append({
                "mode": mode, "signal": entry["signal"],
                "accepted": row is not None, "elapsed_s": round(elapsed, 1),
            })

            if row:
                append_row(writer, fh, row)
                generated   += 1
                total_written += 1
                LOGGER.info(
                    "[%s] %d/%d  signal=%s  lang=%s  (%.0fs)",
                    mode, generated, n, entry["signal"], language, elapsed,
                )

    try:
        LOGGER.info("=== Phase 2g: hard positives (need %d) ===", n_hard_positive)
        _run_batch("hard_positive", n_hard_positive)

        LOGGER.info("=== Phase 2g: adversarial negatives (need %d) ===", n_hard_negative)
        _run_batch("adversarial_negative", n_hard_negative)
    finally:
        fh.close()

    LOGGER.info("✓ Wrote %d rows this run to %s", total_written, output_path)

    # Append to attempt log (don't overwrite previous attempts on resume)
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

    accepted_count = sum(1 for a in attempts_log if a["accepted"])
    LOGGER.info(
        "Accept rate this run: %d/%d (%.0f%%)",
        accepted_count, len(attempts_log),
        100 * accepted_count / max(1, len(attempts_log)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
