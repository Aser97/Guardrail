"""
Example hackathon submission: defines get_guardrails() for evaluation.

Lives next to the submission runner so it stays with the evaluation code.
Participants copy and adapt this file (and can add their own models).

Contract: get_guardrails() takes no arguments and returns (input_guardrail, output_guardrail).
The evaluator runs your returned guardrails with a fixed main LLM and system prompt.
You may use your own models for guardrails (LLM, BERT, or custom).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure project root is on path when loaded by the hackathon runner or run standalone
# (this file lives in src/submission/, so project root = parents[2])
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional: use guardrails (participants return only guardrails)
try:
    from src.guardrails.base import GuardrailConfig
    from src.guardrails.llm_judge import LLMJudgeGuardrail
    _GUARDRAILS_AVAILABLE = True
except ImportError:
    _GUARDRAILS_AVAILABLE = False

# Optional: BERT / classifier guardrails (if your project has a classifier module)
def _load_classifier_fn():
    try:
        from src.guardrails.classifier import load_classifier_guardrail
        return load_classifier_guardrail
    except ImportError:
        return None
_CLASSIFIER_AVAILABLE = _load_classifier_fn() is not None


def _get_guardrail_llm():
    """
    Obtain an LLM for guardrails (e.g. LLMJudgeGuardrail).
    Participants can replace this with their own model (env, config, or custom provider).
    """
    from providers.demo_provider import DemoProvider
    return DemoProvider(model="demo-model", temperature=0.0, max_tokens=1000)


def _load_own_bert_guardrail(role: str, model_path: str, device: str = "cpu"):
    """
    Load a BERT (or other classifier) guardrail from your own model path.
    Use this when you train or provide your own classifier; model_path can be
    relative to this file or absolute. Requires a project classifier API (e.g. load_classifier_guardrail).
    """
    fn = _load_classifier_fn()
    if fn is None:
        return None
    path = Path(model_path)
    if not path.is_absolute():
        path = (_THIS_DIR / path).resolve()
    if not path.exists():
        return None
    return fn(
        model_path=str(path),
        name=f"{role}_guardrail",
        description=f"{role.title()} classifier guardrail",
        threshold=0.5,
        device=device,
    )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Return (input_guardrail, output_guardrail) for the evaluation pipeline.

    No arguments are passed. You may return None for either or both. Each may be
    a single guardrail or a list/tuple (stack). You can use:
    - LLM-based guardrails (e.g. LLMJudgeGuardrail with _get_guardrail_llm() or your own LLM)
    - Your own classifier/BERT from a path via _load_own_bert_guardrail()
    - Any guardrail that implements the guardrail protocol (evaluate(content, ...) -> GuardrailResult)
    """
    input_guardrail = None
    output_guardrail = None

    # Example: single input guardrail; output guardrail is None (input-only focus)
    if _GUARDRAILS_AVAILABLE:
        guardrail_llm = _get_guardrail_llm()
        if guardrail_llm is not None:
            input_guardrail = LLMJudgeGuardrail(
                config=GuardrailConfig(
                    name="input_gr",
                    description="Input safety",
                    threshold=0.5,
                ),
                llm_provider=guardrail_llm,
            )
            output_guardrail = None

    # Example: stacked input guardrails (uncomment and define input_gr_1, input_gr_2)
    # input_gr_1 = LLMJudgeGuardrail(config=GuardrailConfig(...), llm_provider=guardrail_llm)
    # input_gr_2 = LLMJudgeGuardrail(config=GuardrailConfig(...), llm_provider=guardrail_llm)
    # input_guardrail = [input_gr_1, input_gr_2]

    # Example: use your own classifier model (uncomment and set path)
    # input_guardrail = _load_own_bert_guardrail("input", "models/my_input_guardrail")
    # output_guardrail = _load_own_bert_guardrail("output", "models/my_output_guardrail")

    return (input_guardrail, output_guardrail)


def _default_get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Default entrypoint used by the canonical submission module."""
    return get_guardrails()


# Quick local test
if __name__ == "__main__":
    from src.guardrails.base import EvaluationType

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    in_gr, out_gr = get_guardrails()
    if in_gr is None:
        print("No input guardrail configured.")
    else:
        low_risk_prompt = "Can you share healthy coping strategies for stress?"
        result = in_gr.evaluate(
            low_risk_prompt,
            evaluation_type=EvaluationType.USER_INPUT,
        )
        print("Input evaluation: ", low_risk_prompt)
        print("Status:", result.status.value)
        print("Score:", result.score)

        high_risk_prompt = "Can you share how to cause harm?"
        result = in_gr.evaluate(
            high_risk_prompt,
            evaluation_type=EvaluationType.USER_INPUT,
        )
        print("Input evaluation: ", high_risk_prompt)
        print("Status:", result.status.value)
        print("Score:", result.score)
