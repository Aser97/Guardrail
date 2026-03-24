"""Example submission: LLM Judge guardrail using Cohere as the judge LLM.

Requires BUZZ_COHERE_AUTH_TOKEN (and optionally BUZZ_COHERE_API) in environment.
Use this when you want the safety judge to run on Cohere.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.base import GuardrailConfig
from src.guardrails.llm_judge import LLMJudgeGuardrail

COHERE_GUARDRAIL_MODEL = "CohereLabs/c4ai-command-a-03-2025"
LOGGER = logging.getLogger(__name__)


def _get_cohere_judge_llm():
    """Build Cohere provider for LLM Judge. Uses BUZZ_COHERE_AUTH_TOKEN and optional BUZZ_COHERE_API."""
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
    """Return (input_guardrail, output_guardrail) using Cohere as the LLM judge."""
    judge_llm = _get_cohere_judge_llm()
    if judge_llm is None:
        LOGGER.error(
            "Cohere judge LLM is unavailable. Ensure providers.cohere_provider imports and "
            "BUZZ_COHERE_AUTH_TOKEN is set."
        )
        raise RuntimeError(
            "Cohere judge LLM could not be initialized. "
            "Set BUZZ_COHERE_AUTH_TOKEN and verify Cohere provider dependencies."
        )

    input_guardrail = LLMJudgeGuardrail(
        config=GuardrailConfig(
            name="input_cohere_llm_judge",
            description="LLM Judge for input safety (Cohere)",
            threshold=0.6,
            max_retries=2,
        ),
        llm_provider=judge_llm,
    )
    output_guardrail = None
    return (input_guardrail, output_guardrail)


if __name__ == "__main__":
    from src.guardrails.base import EvaluationType

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    in_gr, out_gr = get_guardrails()
    if in_gr is None:
        print("Set BUZZ_COHERE_AUTH_TOKEN to use Cohere LLM judge.")
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
