"""Example submission: stacked input guardrails with mmBERT twice.

Contract: get_guardrails() -> (input_guardrail, output_guardrail)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.classifier import load_classifier_guardrail
from src.submission._runtime_config import resolve_device_from_hackathon


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return stacked input guardrails: [mmBERT_1, mmBERT_2]."""
    model_path1 = _PROJECT_ROOT / "models" / "mmbert_guardrail_demo1" / "mmbert_guardrail_demo"
    if not model_path1.exists():
        raise RuntimeError(
            f"Finetuned mmBERT model not found at {model_path1}. "
            "Train/export the model first, then re-run."
        )

    device = resolve_device_from_hackathon(_PROJECT_ROOT)

    mmbert_guardrail_1 = load_classifier_guardrail(
        model_path=str(model_path1),
        name="input_mmbert_finetuned_1",
        description="Finetuned mmBERT input safety guardrail (stacked #1)",
        threshold=0.30,
        device=device,
    )

    model_path2 = _PROJECT_ROOT / "models" / "mmbert_guardrail_demo2" / "mmbert_guardrail_demo"
    if not model_path2.exists():
        raise RuntimeError(
            f"Finetuned mmBERT model not found at {model_path2}. "
            "Train/export the model first, then re-run."
        )

    mmbert_guardrail_2 = load_classifier_guardrail(
        model_path=str(model_path2),
        name="input_mmbert_finetuned_2",
        description="Finetuned mmBERT input safety guardrail (stacked #2)",
        threshold=0.45,
        device=device,
    )

    input_guardrail = [mmbert_guardrail_1, mmbert_guardrail_2]
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