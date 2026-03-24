"""Example submission: LLM Judge guardrail using OpenAI GPT as the judge LLM.

Use this when you want the safety judge to run on OpenAI GPT.
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

LOGGER = logging.getLogger(__name__)

GPT_GUARDRAIL_MODEL = "openai/gpt-oss-120b"

def _get_gpt_judge_llm():
    """Build GPT provider for LLM Judge. Uses BUZZ_GPT_OSS_AUTH_TOKEN and optional BUZZ_GPT_OSS_API."""
    try:
        from providers.openai_provider import OpenAIProvider
    except ImportError:
        return None
    api_key = os.getenv("BUZZ_GPT_OSS_AUTH_TOKEN")
    if not api_key:
        return None
    base_url = os.getenv("BUZZ_GPT_OSS_API") or None
    return OpenAIProvider(
        base_url=base_url,
        model=GPT_GUARDRAIL_MODEL,
        temperature=0.0,
        max_tokens=1000,
        api_key=api_key,
        verify_ssl=False,
    )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail) using OpenAI GPT as the LLM judge."""
    judge_llm = _get_gpt_judge_llm()
    if judge_llm is None:
        LOGGER.error(
            "OpenAI judge LLM is unavailable. Ensure providers.openai_provider imports and "
            "BUZZ_GPT_OSS_AUTH_TOKEN is set."
        )
        raise RuntimeError(
            "OpenAI GPT judge LLM could not be initialized. "
            "Set BUZZ_GPT_OSS_AUTH_TOKEN and verify OpenAI provider dependencies."
        )

    input_guardrail = LLMJudgeGuardrail(
        config=GuardrailConfig(
            name="input_openai_llm_judge",
            description="LLM Judge for input safety (OpenAI GPT)",
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
        print("Set BUZZ_GPT_OSS_AUTH_TOKEN to use OpenAI GPT LLM judge.")
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