"""Finetunable classifier guardrail using Hugging Face Transformers.

Same contract as LLMJudgeGuardrail: evaluate(content, context, evaluation_type) -> GuardrailResult.
Compatible with the chat pipeline and guardrails_loader (type: "finetunable" or "bert").
Uses only the Hugging Face transformers library (no sklearn).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import (
    BaseGuardrail,
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
    EvaluationType,
)

# Artifact written by training script (transformers save_pretrained)
CONFIG_FILENAME = "guardrail_config.json"
LOGGER = logging.getLogger(__name__)


def _env_bool(name: str) -> bool:
    """Return whether an environment variable is set (without exposing value)."""
    return bool(os.getenv(name))


def _looks_like_local_path(model_path: str) -> bool:
    """Heuristic: explicit filesystem-like paths should be treated as local paths."""
    return (
        model_path.startswith("/")
        or model_path.startswith("./")
        or model_path.startswith("../")
        or model_path.startswith("~")
    )


def _has_any_file(path: Path, candidates: tuple[str, ...]) -> bool:
    """Return True when any candidate file exists in path."""
    return any((path / filename).exists() for filename in candidates)


def _has_any_glob(path: Path, patterns: tuple[str, ...]) -> bool:
    """Return True when any glob pattern matches at least one file."""
    for pattern in patterns:
        if any(path.glob(pattern)):
            return True
    return False


def _validate_local_model_dir(model_dir: Path) -> None:
    """
    Validate local model directory is plausibly loadable.
    This check stays generic: it accepts common HF layouts and many custom PyTorch checkpoints.
    It only fails when the directory does not look like a model directory at all.
    """
    # Common Hugging Face artifacts (at least one is enough to consider directory plausible).
    hf_marker_files = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "preprocessor_config.json",
        "feature_extractor_config.json",
    )
    # Generic training checkpoints often seen across teams/frameworks.
    checkpoint_globs = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.ckpt",
    )

    has_hf_markers = _has_any_file(model_dir, hf_marker_files)
    has_generic_checkpoints = _has_any_glob(model_dir, checkpoint_globs)
    if has_hf_markers or has_generic_checkpoints:
        # Soft warning for likely non-HF raw checkpoints: pipeline may still fail later, but keep this generic.
        has_pt_checkpoints = _has_any_glob(model_dir, ("*.pt", "*.pth", "*.ckpt"))
        if has_pt_checkpoints and not has_hf_markers:
            LOGGER.warning(
                "Model directory has generic checkpoint files but no standard HF metadata. "
                "If loading fails, export with save_pretrained() or include tokenizer/config artifacts. "
                "model_dir=%s",
                model_dir,
            )
        LOGGER.info(
            "Validated local model directory | model_dir=%s has_hf_markers=%s has_generic_checkpoints=%s",
            model_dir,
            has_hf_markers,
            has_generic_checkpoints,
        )
        return

    present_files = sorted(p.name for p in model_dir.iterdir())
    message = (
        f"Local classifier directory does not appear to contain model artifacts: {model_dir}. "
        "Expected at least one known model/checkpoint file "
        "(e.g. config.json, model.safetensors, pytorch_model.bin, *.pt, *.pth, *.ckpt)."
    )
    LOGGER.error("%s Present files: %s", message, present_files)
    raise ValueError(message)


def _model_load_hint(exc: Exception, model_id: str) -> str:
    """Generate actionable hint text for common transformers load failures."""
    msg = str(exc).lower()
    if any(token in msg for token in ("401", "403", "unauthorized", "forbidden", "token")):
        return (
            "Authentication/permissions issue while accessing the model. "
            "If this is a private or gated model, set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
            "with access to the repository."
        )
    if any(
        token in msg
        for token in (
            "repository not found",
            "not a valid model identifier",
            "revision not found",
            "entry not found",
        )
    ):
        return (
            f"Model id/path '{model_id}' was not found. Verify spelling, repository visibility, "
            "or use a valid local model directory."
        )
    if any(token in msg for token in ("gated", "access to model", "accept the terms")):
        return (
            "Model appears gated. Accept the model terms on Hugging Face and ensure your token "
            "has permission."
        )
    if any(
        token in msg
        for token in (
            "connection error",
            "read timed out",
            "max retries exceeded",
            "temporary failure in name resolution",
            "couldn't connect",
        )
    ):
        return (
            "Network issue while downloading model artifacts. Check internet access and retry, "
            "or use a local pretrained directory."
        )
    return (
        "Model loading failed. Check model path/artifacts, transformer compatibility, and "
        "environment dependencies."
    )


def _log_runtime_env(device: str) -> None:
    """Log non-sensitive runtime env diagnostics useful for prediction/metrics debugging."""
    LOGGER.info(
        "Classifier runtime env | requested_device=%s CUDA_VISIBLE_DEVICES_set=%s HF_TOKEN_set=%s TRANSFORMERS_CACHE_set=%s",
        device,
        _env_bool("CUDA_VISIBLE_DEVICES"),
        _env_bool("HF_TOKEN") or _env_bool("HUGGINGFACE_HUB_TOKEN"),
        _env_bool("TRANSFORMERS_CACHE"),
    )


def _is_gpu_oom_error(exc: Exception) -> bool:
    """Detect common CUDA/GPU out-of-memory error patterns."""
    msg = str(exc).lower()
    patterns = (
        "cuda out of memory",
        "out of memory",
        "cudnn_status_alloc_failed",
        "cuda error: out of memory",
        "hip out of memory",
        "insufficient memory",
    )
    return any(p in msg for p in patterns)


def _log_cuda_memory_snapshot(prefix: str) -> None:
    """Best-effort GPU memory snapshot for debugging model load/eval failures."""
    try:
        import torch

        if not torch.cuda.is_available():
            LOGGER.warning("%s | CUDA not available while collecting memory snapshot", prefix)
            return
        device_idx = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        LOGGER.error(
            "%s | cuda_device=%d free_gb=%.2f total_gb=%.2f allocated_gb=%.2f reserved_gb=%.2f",
            prefix,
            device_idx,
            free_bytes / (1024**3),
            total_bytes / (1024**3),
            allocated / (1024**3),
            reserved / (1024**3),
        )
    except Exception:
        LOGGER.exception("%s | Failed to collect CUDA memory snapshot", prefix)


def _load_transformers_pipeline(model_path: str, device: str = "cpu"):
    """
    Load a Hugging Face text-classification pipeline.

    Supports:
    - Local directory: path to a model saved with save_pretrained() (e.g. from
      src.guardrails.train_classifier_guardrail). BERT and other HF models are saved
      in this format.
    - Hugging Face Hub model id: e.g. "bert-base-uncased", "roberta-base",
      "distilbert-base-uncased", or any model id that has a sequence
      classification head (num_labels=2 for binary low_risk/high_risk).
    """
    LOGGER.info("Loading classifier model from path/id: %s", model_path)
    LOGGER.info("Loading transformers pipeline | model_path=%s device=%s", model_path, device)
    if device != "cuda":
        LOGGER.warning(
            "Classifier is configured to run on CPU (device=%s). Inference may be slower for large models. If you want to run on GPU, set it in hackathon.json 'needs_gpu' to true.",
            device,
        )
    _log_runtime_env(device)
    try:
        from transformers import pipeline
    except ImportError:
        LOGGER.exception("transformers import failed while loading classifier pipeline")
        raise ImportError(
            "transformers is required for the classifier guardrail. "
            "Install with: pip install transformers torch"
        )
    path = Path(model_path).expanduser()
    explicit_local = _looks_like_local_path(model_path)
    if explicit_local and not path.exists():
        LOGGER.error("Classifier local model path does not exist: %s", path.resolve())
        raise FileNotFoundError(
            f"Classifier model path does not exist: {path.resolve()}. "
            "Update model_path to a valid local directory or use a valid Hugging Face model id."
        )
    if explicit_local and path.exists() and not path.is_dir():
        LOGGER.error("Classifier local model path is not a directory: %s", path.resolve())
        raise NotADirectoryError(
            f"Classifier model path is not a directory: {path.resolve()}. "
            "Expected a directory created by save_pretrained()."
        )

    load_from_local = path.exists() and path.is_dir()
    if load_from_local:
        _validate_local_model_dir(path.resolve())
    model_id = str(path.resolve()) if load_from_local else model_path
    if load_from_local:
        LOGGER.info("Classifier model source resolved to local path: %s", model_id)
    else:
        LOGGER.info("Classifier model source resolved to Hub/model id: %s", model_id)
    LOGGER.info(
        "Resolved model source | load_from_local=%s model_id=%s",
        load_from_local,
        model_id,
    )
    try:
        pipe = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            device=0 if device == "cuda" else -1,  # -1 = CPU
            top_k=None,  # return all class scores
        )
    except Exception as exc:
        if device == "cuda" and _is_gpu_oom_error(exc):
            _log_cuda_memory_snapshot("GPU OOM while loading classifier model")
            LOGGER.error(
                "Failed to load classifier model on GPU: model likely too large for available VRAM. "
                "Try using a smaller model, running on CPU (device='cpu'), or freeing GPU memory."
            )
            raise RuntimeError(
                "Failed to load classifier model on GPU because it appears too large for available memory. "
                "Try a smaller model, set device='cpu', or free GPU memory and retry."
            ) from exc
        LOGGER.exception("Transformers pipeline initialization failed")
        hint = _model_load_hint(exc, model_id)
        LOGGER.error("Classifier model load hint: %s", hint)
        source_hint = (
            f"local model path '{path.resolve()}'"
            if load_from_local or explicit_local
            else f"Hugging Face model id '{model_id}'"
        )
        raise RuntimeError(
            f"Unable to load classifier model from {source_hint}. "
            f"Hint: {hint} "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    resolved_model_name = getattr(getattr(pipe, "model", None), "name_or_path", None)
    resolved_tokenizer_name = getattr(getattr(pipe, "tokenizer", None), "name_or_path", None)
    LOGGER.info(
        "Transformers pipeline loaded successfully | requested_model=%s resolved_model=%s resolved_tokenizer=%s",
        model_id,
        resolved_model_name,
        resolved_tokenizer_name,
    )
    return pipe


def _load_config(model_path: str) -> Dict[str, Any]:
    """Load optional guardrail_config.json from model directory (local path only)."""
    LOGGER.info("Attempting to load classifier config from model path: %s", model_path)
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        LOGGER.debug("Model path is not a local directory; skipping %s load", CONFIG_FILENAME)
        return {}
    config_path = path / CONFIG_FILENAME
    if not config_path.exists():
        LOGGER.info("No %s found under model path", CONFIG_FILENAME)
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
            LOGGER.info("Loaded classifier config artifact: %s", config_path)
            return cfg
    except Exception:
        LOGGER.exception("Failed to parse classifier config artifact: %s", config_path)
        return {}


def _predict_risk(pipe: Any, text: str, id2label: Optional[Dict[int, str]] = None) -> float:
    """
    Return risk score in [0, 1] (higher = more high_risk). Used for blocking: block when score >= threshold.
    Resolves P(high_risk) from the model's id2label (e.g. "high_risk" or LABEL_1).
    Training script uses 0=low_risk, 1=high_risk; we always return the score for the high_risk class.
    """
    if not text or not text.strip():
        LOGGER.error("Empty text received; refusing to return default score")
        raise ValueError("Empty text is not allowed for classification")
    result = pipe(
        text.strip(),
        truncation=True,
        max_length=512,
        padding=True,
        return_all_scores=True,
    )
    if not result or not result[0]:
        LOGGER.error("Classifier pipeline returned empty result; refusing to return default score")
        raise RuntimeError("Classifier pipeline returned empty result")
    preds = result[0] if isinstance(result[0], list) else [result[0]]
    # Build set of names that mean "high_risk" (training uses 0=low_risk, 1=high_risk)
    high_risk_label_names = {"high_risk"}
    high_risk_index = 1  # default from training script
    if id2label:
        LOGGER.debug("Using id2label mapping to resolve high_risk class: %s", id2label)
        for idx, name in id2label.items():
            if name and str(name).lower() == "high_risk":
                high_risk_label_names.add(name)
                high_risk_index = int(idx)
                break
        if 1 in id2label and id2label[1]:
            high_risk_label_names.add(id2label[1])
    # Also match HF pipeline style "LABEL_0" / "LABEL_1"
    high_risk_label_names.add(f"LABEL_{high_risk_index}")
    # Find score for the high_risk class
    for p in preds:
        label = p.get("label", "")
        if "score" not in p:
            LOGGER.error("Prediction missing 'score' field: %s", p)
            raise RuntimeError(f"Classifier prediction missing 'score' field: {p}")
        score = float(p["score"])
        if label in high_risk_label_names:
            LOGGER.debug("Matched high_risk label=%s score=%.6f", label, score)
            return score
        if not id2label and ("1" in str(label) or str(label).lower() == "high_risk"):
            LOGGER.debug("Matched heuristic high_risk label=%s score=%.6f", label, score)
            return score
    # No fallback: if we cannot find high_risk label, raise error
    LOGGER.error(
        "Could not map any label to high_risk; labels found: %s",
        [p.get("label") for p in preds],
    )
    raise RuntimeError(
        f"Could not determine high_risk score from classifier output. Labels: {[p.get('label') for p in preds]}"
    )


class ClassifierGuardrail(BaseGuardrail):
    """
    Guardrail that uses a Hugging Face Transformers model to score content.

    Supports any HF model for text/sequence classification (e.g. BERT, RoBERTa,
    DistilBERT, ALBERT). Score convention matches LLM judge: higher score = more
    high_risk/risky. FAIL when score >= config.threshold. Compatible with
    guardrail stacks and evaluator runtime.
    """

    def __init__(
        self,
        config: GuardrailConfig,
        pipeline: Any,
        device: str = "cpu",
        id2label: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            config: Guardrail configuration (threshold, etc.).
            pipeline: Hugging Face text-classification pipeline (transformers).
            device: "cpu" or "cuda" for inference.
            id2label: Optional mapping from label index to name (for P(high_risk) lookup).
        """
        super().__init__(config)
        self.pipeline = pipeline
        self.device = device
        self.id2label = id2label or {}
        LOGGER.info(
            "ClassifierGuardrail ready | name=%s device=%s threshold=%.4f id2label_keys=%s",
            self.config.name,
            self.device,
            self.config.threshold,
            sorted(self.id2label.keys()),
        )

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        start_time = time.time()
        evaluation_type_value = (
            evaluation_type.value if isinstance(evaluation_type, EvaluationType) else str(evaluation_type)
        )
        try:
            if not isinstance(content, str):
                LOGGER.error(
                    "Input dataset schema not respected in classifier evaluate: expected 'content' as str, got %s",
                    type(content).__name__,
                )
                raise ValueError(
                    "Input dataset schema not respected: each row must provide text content as a string."
                )
            if context is not None and not isinstance(context, dict):
                LOGGER.error(
                    "Input dataset schema not respected in classifier evaluate: expected 'context' as dict or None, got %s",
                    type(context).__name__,
                )
                raise ValueError(
                    "Input dataset schema not respected: context must be a dictionary when provided."
                )
            content_preview = (content or "").strip().replace("\n", " ")[:120]
            LOGGER.debug(
                "Classifier evaluate start | guardrail=%s evaluation_type=%s content_chars=%d preview=%r",
                self.config.name,
                evaluation_type_value,
                len((content or "").strip()),
                content_preview,
            )
            score = _predict_risk(self.pipeline, content, self.id2label)
            status = GuardrailStatus.FAIL if score >= self.config.threshold else GuardrailStatus.PASS
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.info(
                "Classifier evaluate success | guardrail=%s status=%s score=%.6f threshold=%.4f latency_ms=%.3f",
                self.config.name,
                status.value,
                score,
                self.config.threshold,
                latency_ms,
            )
            return GuardrailResult(
                status=status,
                score=score,
                reasoning=None,
                metadata={
                    "evaluation_type": evaluation_type_value,
                    "model_type": "transformers",
                },
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            is_gpu_oom = self.device == "cuda" and _is_gpu_oom_error(e)
            if is_gpu_oom:
                _log_cuda_memory_snapshot("GPU OOM during classifier evaluate")
                LOGGER.error(
                    "Classifier inference failed due to GPU memory pressure. "
                    "Model/input may be too large for current VRAM; consider CPU fallback or reducing model size."
                )
            raise RuntimeError(
                f"Classifier evaluation failed for guardrail '{self.config.name}'. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e


def load_classifier_guardrail(
    model_path: str,
    name: str,
    description: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> BaseGuardrail:
    """
    Load a finetunable classifier guardrail from a Hugging Face model.

    model_path can be:
    - A local directory: saved with the training script (save_pretrained).
      Example: "models/my_guardrail" (BERT or any HF model fine-tuned and saved).
    - A Hugging Face Hub model id: e.g. "bert-base-uncased", "roberta-base",
      "distilbert-base-uncased". The model must support sequence classification
      (binary: num_labels=2 for low_risk/high_risk). Fine-tune with the training
      script for best results.

    Compatible with the evaluator runtime: use as input_guardrail or output_guardrail
    (single or in a stack with LLMJudgeGuardrail).

    Args:
        model_path: Local path to saved model dir, or HF model id (e.g. bert-base-uncased).
        name: Guardrail name (e.g. "input_guardrail").
        description: Short description for config.
        threshold: Score >= threshold -> FAIL (higher score = more high_risk).
        device: "cpu" or "cuda" for inference.

    Returns:
        ClassifierGuardrail instance usable in the pipeline and loader.
    """
    LOGGER.info(
        "Loading classifier guardrail | model_path=%s name=%s threshold=%.4f device=%s",
        model_path,
        name,
        threshold,
        device,
    )
    try:
        pipe = _load_transformers_pipeline(model_path, device)
    except Exception:
        LOGGER.exception(
            "Classifier guardrail load failed before initialization | model_path=%s name=%s device=%s",
            model_path,
            name,
            device,
        )
        raise
    meta = _load_config(model_path)
    if meta:
        LOGGER.info(
            "Applying classifier artifact config overrides | threshold=%s keys=%s",
            meta.get("threshold"),
            sorted(meta.keys()),
        )
        threshold = meta.get("threshold", threshold)
    config = GuardrailConfig(
        name=name,
        description=description,
        threshold=threshold,
    )
    id2label = getattr(pipe.model.config, "id2label", None)
    if id2label is not None and isinstance(id2label, dict):
        normalized_id2label: Dict[int, str] = {}
        for k, v in id2label.items():
            try:
                normalized_id2label[int(k)] = v
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Skipping non-integer id2label key from model config | key=%r value=%r",
                    k,
                    v,
                )
        id2label = normalized_id2label
    else:
        id2label = {}
    LOGGER.info(
        "Classifier guardrail loaded | name=%s final_threshold=%.4f id2label=%s",
        name,
        threshold,
        id2label,
    )
    return ClassifierGuardrail(config=config, pipeline=pipe, device=device, id2label=id2label)
