"""
project/src/submission/submission.py
Team 027 — Mental Health Safety Sandbox Hackathon

Submission entry point.  The evaluator imports this module and calls
get_guardrails() exactly once to obtain the input guardrail.

Model  : Qwen2.5-7B-Instruct fine-tuned with LoRA (r=16, α=32) as a
         9-head multi-label sequence classifier.
         Sigmoid activation + BCEWithLogitsLoss during training.

         Two-phase scoring pipeline:
           Phase 1 — Qwen backbone: produces 9 signal probabilities p_1…p_9.
           Phase 2 — Logistic regression aggregation head: computes
             z = w · p + b  (dot product of 9 weights + bias),
             score = sigmoid(z) = P(high_risk).
         The aggregation head is trained on the validation split (honest
         probabilities) and stored in aggregation_head.json. It learns
         which signal combinations actually predict the label, correctly
         suppressing third_party_concern and testing from auto-escalating
         and up-weighting correlated risk signals. Threshold = 0.5.

Signals: burden_language · finality_language · escape_framing ·
         hopelessness · active_self_harm · immediate_safety ·
         self_image_crisis · third_party_concern · testing

Path   : project/models/mhs_guardrail/
         ├── adapter_config.json          (LoRA config, saved by PEFT)
         ├── adapter_model.safetensors    (LoRA delta weights)
         ├── classifier_head.pt           (linear head state_dict)
         ├── aggregation_head.json        (LR weights + bias, Phase 2)
         └── tokenizer/                   (tokenizer files)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Signal order must match the training script's label ordering exactly
_SIGNALS: list[str] = [
    "burden_language",
    "finality_language",
    "escape_framing",
    "hopelessness",
    "active_self_harm",
    "immediate_safety",
    "self_image_crisis",
    "third_party_concern",
    "testing",
]

_THRESHOLD   = 0.7075    # aggregation head decision boundary (natural LR operating point)
_MAX_LENGTH  = 4096   # hard ceiling; truncation_side="left" → keep most recent turns
_BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"


# ──────────────────────────────────────────────────────────────────────────────
# Custom guardrail class
# ──────────────────────────────────────────────────────────────────────────────

class MHSGuardrail:
    """
    Qwen2.5-7B-Instruct + LoRA multi-label mental-health safety classifier
    with a logistic regression aggregation head (Phase 2).

    Satisfies the BaseGuardrail / GuardrailProtocol interface:
        evaluate(content, context, evaluation_type) → GuardrailResult

    Score convention (per base.py docstring): higher = more risky.
    score = sigmoid(w · probs + b) — logistic regression aggregation trained
    on validation split signal probabilities. Falls back to max(probs) if
    aggregation_head.json is not found in the model directory.
    """

    def __init__(
        self,
        config,           # GuardrailConfig
        model_path: Path,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoTokenizer
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        self.config    = config
        self.device    = device
        self.threshold = config.threshold

        tok_path = model_path / "tokenizer"
        if not tok_path.exists():
            tok_path = model_path  # fallback: tokenizer saved at root

        LOGGER.info("Loading tokenizer from %s", tok_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tok_path),
            truncation_side="left",    # drop the beginning; keep recent turns
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if "cuda" in device else torch.float32

        LOGGER.info("Loading Qwen2.5-7B-Instruct base model (dtype=%s)…", dtype)
        base = AutoModelForCausalLM.from_pretrained(
            model_path / "base_model",
            torch_dtype=dtype,
            device_map=device,
        )

        LOGGER.info("Applying LoRA adapters from %s", model_path)
        self._backbone = PeftModel.from_pretrained(base, str(model_path))
        self._backbone.eval()

        # Classification head: linear(hidden_size → 9)
        hidden_size = base.config.hidden_size
        self._head = torch.nn.Linear(hidden_size, len(_SIGNALS), bias=True)
        head_path  = model_path / "classifier_head.pt"
        self._head.load_state_dict(
            torch.load(str(head_path), map_location=device)
        )
        self._head = self._head.to(device=device, dtype=dtype)
        self._head.eval()

        # Aggregation head: logistic regression (Phase 2)
        # Trained on val-split honest probabilities; stored as JSON.
        # Suppresses third_party_concern / testing from auto-escalating;
        # weights correlated risk signals appropriately.
        import json as _json
        agg_path = model_path / "aggregation_head.json"
        if agg_path.exists():
            agg = _json.loads(agg_path.read_text())
            self._agg_w: list[float] = agg["weights"]   # 9 floats
            self._agg_b: float       = float(agg["bias"])
            LOGGER.info(
                "Aggregation head loaded from %s | bias=%.4f",
                agg_path, self._agg_b,
            )
        else:
            self._agg_w = None  # type: ignore[assignment]
            self._agg_b = None  # type: ignore[assignment]
            LOGGER.warning(
                "aggregation_head.json not found in %s — "
                "falling back to max(probs) aggregation. "
                "Run scripts/calibrate_threshold.py to generate it.",
                model_path,
            )

        LOGGER.info(
            "MHS guardrail ready | device=%s threshold=%.2f signals=%d agg_head=%s",
            device, self.threshold, len(_SIGNALS),
            "logistic_regression" if self._agg_w is not None else "max(probs)",
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type=None,
    ):
        import torch
        from src.guardrails.base import GuardrailResult, GuardrailStatus

        t0 = time.perf_counter()
        try:
            enc = self.tokenizer(
                content,
                return_tensors="pt",
                max_length=_MAX_LENGTH,
                truncation=True,
                padding=False,
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self._backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Use the last token's hidden state from the final layer
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # (1, hidden)
                logits = self._head(last_hidden)                    # (1, 9)
                probs  = torch.sigmoid(logits).squeeze(0).tolist()  # [9]

            import math as _math
            signal_scores = dict(zip(_SIGNALS, probs))

            # Phase 2: logistic regression aggregation head
            # Learned which signal combinations predict the label, including
            # suppression of third_party_concern / testing false positives.
            if self._agg_w is not None:
                z             = sum(w * p for w, p in zip(self._agg_w, probs)) + self._agg_b
                overall_score = 1.0 / (1.0 + _math.exp(-z))   # sigmoid → P(high_risk)
                agg_method    = "logistic_regression"
            else:
                # Fallback: max aggregation (Phase 1 behaviour)
                overall_score = max(probs)
                agg_method    = "max"

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Decision based on aggregation head score (not per-signal threshold)
            triggered = [s for s, p in signal_scores.items() if p >= self.threshold]
            if overall_score >= self.threshold:
                status    = GuardrailStatus.FAIL
                reasoning = (
                    f"Risk score {overall_score:.3f} ≥ {self.threshold:.2f} "
                    f"(agg={agg_method}). "
                    f"Active signals: {', '.join(triggered) if triggered else 'none above per-signal threshold'}."
                )
            else:
                status    = GuardrailStatus.PASS
                reasoning = (
                    f"Risk score {overall_score:.3f} < {self.threshold:.2f} "
                    f"(agg={agg_method}). No high-risk decision."
                )

            return GuardrailResult(
                status=status,
                score=overall_score,
                reasoning=reasoning,
                metadata={
                    "signal_scores":  signal_scores,
                    "triggered":      triggered,
                    "threshold":      self.threshold,
                    "agg_method":     agg_method,
                    "latency_ms":     latency_ms,
                },
            )

        except Exception as exc:
            LOGGER.error("MHS guardrail evaluation error: %s", exc, exc_info=True)
            from src.guardrails.base import GuardrailResult, GuardrailStatus
            return GuardrailResult(
                status=GuardrailStatus.ERROR,
                reasoning=f"Evaluation error: {exc}",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Return (input_guardrail, output_guardrail).

    The input guardrail intercepts every user message BEFORE it reaches the LLM.
    Internally it runs a two-phase pipeline:
      Phase 1 — Qwen2.5-7B + LoRA produces 9 signal probabilities.
      Phase 2 — logistic regression aggregation head converts them to P(high_risk).
    Output guardrail is not required — return None for it.
    """
    from src.guardrails.base import GuardrailConfig

    LOGGER.info("Loading MHS guardrail (team_027)")

    model_path = _PROJECT_ROOT / "models" / "mhs_guardrail"

    if not model_path.exists():
        LOGGER.warning(
            "Model directory not found: %s — returning (None, None). "
            "Run scripts/train_guardrail.sh to train the model first.",
            model_path,
        )
        return (None, None)

    try:
        from src.submission._runtime_config import resolve_device_from_hackathon
        device = resolve_device_from_hackathon(_PROJECT_ROOT)
    except Exception as exc:
        LOGGER.warning("Could not resolve device (%s) — defaulting to cpu.", exc)
        device = "cpu"

    config = GuardrailConfig(
        name="mhs_input_guardrail",
        description=(
            "Qwen2.5-7B-Instruct + LoRA 9-head multi-label mental-health safety "
            "classifier with logistic regression aggregation head (Phase 2). "
            "Trained on synthetic Canadian EN/FR distress conversations. "
            "Phase 1 produces 9 signal probabilities; Phase 2 combines them via "
            "sigmoid(w·p + b) → P(high_risk). Flags inputs where P(high_risk) >= 0.5. "
            "Signals: burden_language, finality_language, escape_framing, "
            "hopelessness, active_self_harm, immediate_safety, "
            "self_image_crisis, third_party_concern, testing."
        ),
        threshold=_THRESHOLD,
    )

    try:
        guardrail = MHSGuardrail(config=config, model_path=model_path, device=device)
    except Exception as exc:
        LOGGER.error("Failed to load MHS guardrail: %s", exc, exc_info=True)
        return (None, None)

    LOGGER.info(
        "MHS guardrail loaded | device=%s threshold=%.2f model_path=%s",
        device, _THRESHOLD, model_path,
    )
    return (guardrail, None)
