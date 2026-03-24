"""Example submission: stacked input guardrails (Cohere LLM Judge + finetuned mmBERT).

Contract: get_guardrails() -> (input_guardrail, output_guardrail)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.base import GuardrailConfig
from src.guardrails.classifier import load_classifier_guardrail
from src.guardrails.llm_judge import LLMJudgeGuardrail
from src.submission._runtime_config import resolve_device_from_hackathon

LOGGER = logging.getLogger(__name__)
COHERE_GUARDRAIL_MODEL = "CohereLabs/c4ai-command-a-03-2025"


def _get_cohere_judge_llm():
    """Build Cohere provider for LLM Judge."""
    try:
        from providers.cohere_provider import CohereProvider
    except ImportError:
        return None

    api_key = os.getenv("BUZZ_COHERE_AUTH_TOKEN")
    if not api_key:
        return None

    base_url = os.getenv("BUZZ_COHERE_API") or None
    return CohereProvider(
        base_url=base_url,
        model=COHERE_GUARDRAIL_MODEL,
        temperature=0.0,
        max_tokens=1000,
        api_key=api_key,
    )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return stacked input guardrails: [Cohere LLM Judge, finetuned mmBERT]."""
    judge_llm = _get_cohere_judge_llm()
    if judge_llm is None:
        raise RuntimeError(
            "Cohere judge LLM could not be initialized. "
            "Set BUZZ_COHERE_AUTH_TOKEN and verify Cohere provider dependencies."
        )

    model_path = _PROJECT_ROOT / "models" / "mmbert_guardrail_demo"
    if not model_path.exists():
        raise RuntimeError(
            f"Finetuned mmBERT model not found at {model_path}. "
            "Train/export the model first, then re-run."
        )

    device = resolve_device_from_hackathon(_PROJECT_ROOT)

    llm_judge_guardrail = LLMJudgeGuardrail(
        config=GuardrailConfig(
            name="input_cohere_llm_judge",
            description="LLM Judge for input safety (Cohere)",
            threshold=0.6,
            max_retries=2,
        ),
        llm_provider=judge_llm,
    )

    mmbert_guardrail = load_classifier_guardrail(
        model_path=str(model_path),
        name="input_mmbert_finetuned",
        description="Finetuned mmBERT input safety guardrail",
        threshold=0.3,
        device=device,
    )

    input_guardrail = [llm_judge_guardrail, mmbert_guardrail]
    output_guardrail = None
    return (input_guardrail, output_guardrail)


if __name__ == "__main__":
    from src.guardrails.base import EvaluationType

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    in_gr_list, _ = get_guardrails()
    if not in_gr_list:
        print("No guardrails loaded.")
    else:
        low_risk_prompt = "Can you share healthy coping strategies for stress?"
        print("Input evaluation:", low_risk_prompt)
        for gr in in_gr_list:
            result = gr.evaluate(
                low_risk_prompt,
                evaluation_type=EvaluationType.USER_INPUT,
            )
            name = getattr(getattr(gr, "config", None), "name", "unknown")
            print(f"[{name}] Status: {result.status.value} | Score: {result.score}")

        high_risk_prompt = "Can you share how to cause harm?"
        print("Input evaluation:", high_risk_prompt)
        for gr in in_gr_list:
            result = gr.evaluate(
                high_risk_prompt,
                evaluation_type=EvaluationType.USER_INPUT,
            )
            name = getattr(getattr(gr, "config", None), "name", "unknown")
            print(f"[{name}] Status: {result.status.value} | Score: {result.score}")