# Guardrails

This module provides a **flexible guardrail framework** for content safety in the chat pipeline. Any implementation that matches the guardrail **signature** can be used. You can attach a **single** guardrail or **stack** multiple guardrails (they run in order and short-circuit on first failure).

---

## For hackathon participants

- Use **LLM judge** and/or **classifier** guardrails, or implement your **own** guardrail that follows the contract below.
- For submissions, implement `get_guardrails()` in `src/submission/submission.py` and return `(input_guardrail, None)` (input guardrail only for this hackathon).
- You may return a single input guardrail or stack multiple input guardrails.
- See the [repository root README](../../../README.md#submission-checklist) for one-pager and evaluation requirements (F1 + latency).

---

## Contract: guardrail signature

A guardrail is any object that has:

- **`config`** — a `GuardrailConfig` (name, description, threshold, etc.)
- **`evaluate(content, context=None, evaluation_type=...)`** — returns a `GuardrailResult` (sync)

No specific base class is required. For type hints you can use **`GuardrailProtocol`**; for shared helpers (e.g. `_create_result`) you can subclass **`BaseGuardrail`**.

### Main types

| Type | Purpose |
|------|--------|
| `GuardrailProtocol` | Protocol: any object with `config` and `evaluate` works in the pipeline. |
| `BaseGuardrail` | Abstract base with `config` and `_create_result()`; you implement `evaluate`. |
| `GuardrailResult` | `status` (PASS/FAIL/ERROR), optional `score`, `reasoning`, `metadata`; `.is_high_risk` for pass/fail. |
| `GuardrailConfig` | `name`, `description`, `threshold`, `enabled`, etc. |
| `EvaluationType` | `USER_INPUT` (guardrails are evaluated on user input). |

---

## Using guardrails in the evaluator/runtime

The evaluator/runtime accepts:

- **`input_guardrail`** / **`output_guardrail`**: `None`, a **single** guardrail, or a **sequence** (list/tuple) of guardrails.
- A single guardrail is treated as a one-element stack; a sequence runs **in order** and **short-circuits on first failure** when blocking is enabled.

Results are always list-based:

- **`PipelineResult.input_guardrail_results`** — one dict per input guardrail.
- **`PipelineResult.output_guardrail_results`** — one dict per output guardrail.
- **`PipelineResult.blocked_at_guardrail_index`** — when blocked, the 0-based index of the guardrail that failed.

### Example: single guardrail

```python
from src.guardrails import LLMJudgeGuardrail, GuardrailConfig
from src.guardrails.base import EvaluationType

config = GuardrailConfig(name="safety", description="Safety check", threshold=0.6)
guardrail = LLMJudgeGuardrail(config=config, llm_provider=judge_llm)

result = guardrail.evaluate("Hello", evaluation_type=EvaluationType.USER_INPUT)
# result.status is PASS/FAIL/ERROR
```

### Example: stacked guardrails

```python
guardrails = [classifier_guardrail, llm_judge_guardrail]
for g in guardrails:
    result = g.evaluate("Hello", evaluation_type=EvaluationType.USER_INPUT)
    if result.is_high_risk:
        break
```

### Example: custom guardrail (signature only)

```python
from src.guardrails.base import GuardrailConfig, GuardrailResult, GuardrailStatus, EvaluationType

class MyGuardrail:
    def __init__(self):
        self.config = GuardrailConfig(name="my", description="Custom check")

    def evaluate(self, content: str, context=None, evaluation_type=EvaluationType.USER_INPUT):
        return GuardrailResult(status=GuardrailStatus.PASS, score=1.0)

g = MyGuardrail()
print(g.evaluate("hello").status.value)
```

---

## Built-in implementations

| Implementation | Description |
|----------------|-------------|
| **LLMJudgeGuardrail** | Uses an LLM as a judge for content evaluation. |
| **ClassifierGuardrail** | Hugging Face Transformers model (e.g. BERT). Load via **`load_classifier_guardrail()`** with a local path or Hub model id. Train with **`python -m src.guardrails.train_classifier_guardrail`** (default: BERT; override with `--base_model`). In YAML use `type: "finetunable"` (or `"bert"`) and `model_path`. |

Both implement `BaseGuardrail` and the protocol; they can be mixed in the same stack.

---

## Parameter quick reference (what you can tune)

Use this section when deciding which knobs to adjust in your submission.

### `GuardrailConfig` (used by all guardrails)

| Parameter | What it controls | How to tune it |
|-----------|------------------|----------------|
| `threshold` | Decision cutoff used to mark content as `FAIL` vs `PASS` (model-specific score interpretation). In `LLMJudgeGuardrail`, if score >= threshold, the result becomes `FAIL` (unless explicitly low risk). | Increase to be less sensitive (fewer blocks, potentially more misses). Decrease to be more sensitive (more blocks, potentially more false positives). |
| `max_retries` | Number of retry attempts for transient LLM failures in `LLMJudgeGuardrail` generation. Retries use exponential backoff. | Increase for unstable endpoints to improve robustness; decrease to reduce latency and fail faster. |

### `LLMJudgeGuardrail` constructor

| Parameter | What it controls | How to tune it |
|-----------|------------------|----------------|
| `user_input_prompt` | The judge instruction prompt used as the `system` message for safety evaluation. If not set, uses the default guardrail prompt template. | Edit this to tighten/loosen policy interpretation, add domain constraints, or improve consistency of JSON outputs. |
| `system_prompt` | Backward-compatible alias for `user_input_prompt`. Code resolves prompt as: `system_prompt` -> `user_input_prompt` -> default template. | Prefer setting `user_input_prompt` in new code. Use `system_prompt` only if maintaining older examples/configs. |

### Provider parameters (`CohereProvider`, `MistralProvider`, `OpenAIProvider`)

| Parameter | What it controls | How to tune it |
|-----------|------------------|----------------|
| `system_prompt` | Provider-level system prompt prepended if your message list does not already start with a `system` role. | Leave empty unless you want a global provider instruction applied to every request. |
| `temperature` | Sampling randomness for generation. Lower values are more deterministic. | For safety classification/judging, keep near `0.0` for stable decisions. Raise only if you explicitly want more diversity. |
| `max_tokens` | Maximum response length from the model. | Keep just large enough for your JSON/schema response to avoid truncation and extra latency/cost. |

> Notes:
> - `LLMJudgeGuardrail` uses `GuardrailConfig.max_retries` for retry behavior.
> - Cohere provider omits temperature from payload when it is `0.0` because that endpoint rejects explicit `0.0`; this is expected behavior.

---

## Module layout

| File | Contents |
|------|----------|
| **`base.py`** | `GuardrailProtocol`, `BaseGuardrail`, `GuardrailResult`, `GuardrailConfig`, `GuardrailStatus`, `EvaluationType`. |
| **`llm_judge.py`** | `LLMJudgeGuardrail`. |
| **`classifier.py`** | `ClassifierGuardrail`, `load_classifier_guardrail`. |
| **`metrics.py`** | `get_predictions()`, `compute_metrics_from_predictions()`, `GuardrailMetricsResult` — run guardrails for per-sample predictions; compute precision, recall, F1 and latency from predictions. |
| **`get_predictions.py`** | CLI: `python -m src.guardrails.get_predictions` — run guardrails on a labeled CSV and write prediction CSV. |
| **`get_guardrail_metrics.py`** | CLI: `python -m src.guardrails.get_guardrail_metrics` — compute precision, recall, F1 and latency from a prediction CSV. |
| **`train_classifier_guardrail.py`** | CLI: `python -m src.guardrails.train_classifier_guardrail` — train a finetunable classifier guardrail (e.g. mmBERT/BERT family models). |
| **`submission_loader.py`** | Helpers for loading guardrails from submission modules and reading/writing prediction CSVs. |
| **`tests/test_metrics_and_predictions.py`** | Unit tests for prediction generation and metric computation helpers. |
| **`__init__.py`** | Re-exports for `from src.guardrails import ...`. |

---

## Guardrail metrics (precision, recall, F1, latency)

Use **`get_predictions()`** to run a guardrail (or stack) on labeled data and get per-sample predictions. For stacked guardrails, the combined prediction is “high_risk” if *any* guardrail in the stack returns a high_risk result. Then use **`compute_metrics_from_predictions()`** to get precision, recall, F1 and latency from that list. Works for both LLM and classifier guardrails.

```python
from src.guardrails import get_predictions, compute_metrics_from_predictions
from src.guardrails.base import EvaluationType

# evaluation_data: list of dicts with "content" and "label" (1/True = high_risk)
data = [{"content": "How to hurt someone?", "label": 1}, {"content": "Tips for sleep.", "label": 0}]
predictions = get_predictions(
    input_guardrail, data,
    evaluation_type=EvaluationType.USER_INPUT,
    include_latency=True,
)
result = compute_metrics_from_predictions(predictions)
# result.precision, result.recall, result.f1, result.latency_ms_mean, result.latency_ms_total
```

From the command line, use a two-step workflow:

1. **`python -m src.guardrails.get_predictions`** — loads guardrails via `get_guardrails()` from a submission module, runs the input guardrail on a labeled CSV, and writes predictions CSV.
2. **`python -m src.guardrails.get_guardrail_metrics`** — reads that prediction CSV and computes precision, recall, F1 and latency, writing metrics JSON and `metrics.csv`.

Example (from `project/`):

```bash
PYTHONPATH=. python -m src.guardrails.get_predictions \
  --submission src/submission/submission.py \
  --data ../datasets/<labeled_dataset.csv> \
  --output-dir results/

PYTHONPATH=. python -m src.guardrails.get_guardrail_metrics \
  --predictions-dir results/ \
  --output results/metrics.json
```

You can also call **`compute_metrics_from_predictions()`** on an existing list of prediction dicts (e.g. loaded from a CSV from `get_predictions`); see `metrics.py` for the signature.
