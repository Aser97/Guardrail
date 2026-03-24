"""Shared helpers for loading submission guardrails and evaluation data."""

from __future__ import annotations

import csv
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger(__name__)


def describe_guardrail(guardrail: Any) -> str:
    """Return readable guardrail identity for logs."""
    if guardrail is None:
        return "None"
    name = getattr(getattr(guardrail, "config", None), "name", None)
    type_name = type(guardrail).__name__
    return f"{name} ({type_name})" if name else type_name


def load_guardrails_from_module(module_path: Path) -> Tuple[Any, Any]:
    """Load module and return (input_guardrail, output_guardrail) from get_guardrails()."""
    submission_dir = str(module_path.resolve().parent)
    LOGGER.info("Preparing submission module import | path=%s", module_path)
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)
        LOGGER.debug("Added submission dir to sys.path: %s", submission_dir)

    spec = importlib.util.spec_from_file_location("participant_submission", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    LOGGER.info("Importing submission module: %s", module_path)
    try:
        spec.loader.exec_module(module)
    except Exception:
        LOGGER.exception("Submission module import failed: %s", module_path)
        raise
    LOGGER.info("Submission module import completed: %s", module_path)
    get_guardrails = getattr(module, "get_guardrails", None)
    if get_guardrails is None:
        raise RuntimeError(
            "Module must define get_guardrails() -> (input_guardrail, output_guardrail)"
        )
    LOGGER.info("Calling submission get_guardrails()")
    try:
        result = get_guardrails()
    except Exception:
        LOGGER.exception("Submission get_guardrails() raised an exception | module=%s", module_path)
        raise
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        LOGGER.error(
            "Invalid get_guardrails() return shape | module=%s return_type=%s return_repr=%r",
            module_path,
            type(result).__name__,
            result,
        )
        raise TypeError(
            "get_guardrails() must return (input_guardrail, output_guardrail), tuple or list of length 2"
        )
    input_guardrail, _ = result[0], result[1]
    LOGGER.info(
        "Submission get_guardrails() completed | input=%s",
        describe_guardrail(input_guardrail),
    )
    return input_guardrail, _


def load_evaluation_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV with content and label columns. Normalize column names."""
    LOGGER.info("Loading evaluation CSV: %s", csv_path)
    rows: List[Dict[str, Any]] = []
    last_error: Exception | None = None
    content_candidates = ("text", "Text", "content", "prompt")
    label_candidates = ("label", "Label", "is_high_risk")
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            LOGGER.debug("Trying CSV decode with encoding=%s", enc)
            with open(csv_path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    LOGGER.error(
                        "Input dataset schema not respected: CSV has no header row | file=%s",
                        csv_path,
                    )
                    return rows
                header = list(reader.fieldnames)
                LOGGER.info("Detected input CSV columns: %s", header)
                has_content_col = any(col in header for col in content_candidates)
                has_label_col = any(col in header for col in label_candidates)
                if not has_content_col or not has_label_col:
                    raise ValueError(
                        "Input dataset schema not respected: expected at least one content column "
                        f"{content_candidates} and one label column {label_candidates}, "
                        f"but found columns {header}"
                    )
                missing_content_count = 0
                missing_label_count = 0
                for row in reader:
                    content = (
                        row.get("text")
                        or row.get("Text")
                        or row.get("content")
                        or row.get("prompt")
                        or ""
                    ).strip()
                    label = row.get("label") or row.get("Label") or row.get("is_high_risk")
                    if not content:
                        missing_content_count += 1
                    if label is None or str(label).strip() == "":
                        missing_label_count += 1
                    rows.append({"content": content, "label": label})
            LOGGER.info(
                "Loaded %d rows from input CSV (encoding=%s)", len(rows), enc
            )
            if missing_content_count or missing_label_count:
                LOGGER.warning(
                    "Input dataset contains incomplete rows | missing_content=%d missing_label=%d total_rows=%d",
                    missing_content_count,
                    missing_label_count,
                    len(rows),
                )
            return rows
        except UnicodeDecodeError as exc:
            last_error = exc
            rows = []
            LOGGER.warning(
                "Failed decoding CSV with encoding=%s, trying next fallback", enc
            )
            continue
        except ValueError:
            LOGGER.exception("Input dataset schema validation failed")
            raise
    if last_error is not None:
        raise last_error
    return rows


def write_predictions_csv(
    predictions: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    """Write per-sample predictions to CSV (each guardrail column + combined_pred)."""
    if not predictions:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(predictions[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            out = {}
            for k, v in row.items():
                if isinstance(v, bool):
                    out[k] = 1 if v else 0
                else:
                    out[k] = v
            writer.writerow(out)
