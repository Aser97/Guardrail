"""Example submission: finetuned mmBERT guardrail via get_guardrails().

Uses jhu-clsp/mmBERT-base (multilingual BERT from JHU CLSP). Hackathon participants
can copy this file and update:
- model_path to their own finetuned model directory
- threshold/device settings
- whether to guard input, output, or both
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure project root is importable when loaded by runner or notebook
_THIS_DIR = Path(__file__).resolve().parent
# Project dir = parent of src/ (contains scripts/, models/, src/)
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.classifier import load_classifier_guardrail
from src.submission._runtime_config import resolve_device_from_hackathon


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail) for the evaluator."""
    # Update this path to your own finetuned model directory (under project/).
    # Example training command (run from project/ or with full paths):
    # python -m src.guardrails.train_classifier_guardrail \
    #   --data ../datasets/sample_training_data.csv \
    #   --output_dir models/mmbert_guardrail_demo \
    #   --base_model jhu-clsp/mmBERT-base
    model_path = _PROJECT_ROOT / "models" / "mmbert_guardrail_demo"

    if not model_path.exists():
        # Keep notebook/runner usable even before training.
        return (None, None)

    device = resolve_device_from_hackathon(_PROJECT_ROOT)
    input_guardrail = load_classifier_guardrail(
        model_path=str(model_path),
        name="input_mmbert_guardrail",
        description="Finetuned mmBERT (jhu-clsp/mmBERT-base) input safety guardrail",
        threshold=0.3,  # block when P(high_risk) >= 0.3; use 0.5 for fewer false positives
        device=device,
    )

    # Example: input-only setup. Change to another guardrail if needed.
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
        print("No guardrail loaded (train model first).")
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
