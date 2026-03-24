"""Example submission: mmBERT-base from Hub with no finetuning.

Uses jhu-clsp/mmBERT-base directly from Hugging Face Hub. No training step;
the model is loaded as a sequence classifier (classification head is
initialized randomly if the Hub model is encoder-only). Useful for
quick demos or as a baseline before finetuning.

Contract: get_guardrails() -> (input_guardrail, output_guardrail).
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


# Hugging Face Hub model id — no local finetuned checkpoint
# NOTE: This model is downloaded from Hugging Face Hub (~500 MB) on first run.
# If running in a network-restricted environment, download the model manually
# and declare it as an artifact in hackathon.json.
BASE_MODEL_ID = "jhu-clsp/mmBERT-base"


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail). Uses base model from Hub, no finetuning."""
    device = resolve_device_from_hackathon(_PROJECT_ROOT)
    input_guardrail = load_classifier_guardrail(
        model_path=BASE_MODEL_ID,
        name="input_mmbert_base_no_finetuning",
        description="mmBERT-base (jhu-clsp/mmBERT-base from Hub, no finetuning)",
        threshold=0.5,
        device=device,
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
        print("No guardrail loaded.")
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