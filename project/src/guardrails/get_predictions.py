#!/usr/bin/env python3
"""Run input guardrail on a labeled CSV and write per-sample predictions.

Loads guardrails via get_guardrails() from the submission module, runs the
input guardrail on the data, and writes predictions to CSV. Use
get_guardrail_metrics afterward to compute precision, recall, F1 and
latency from that file.

Usage:
    cd project && PYTHONPATH=. python -m src.guardrails.get_predictions \\
        --submission src/submission/submission.py \\
        --data ../datasets/seed_validation_set.csv \\
        --output-dir results/

Writes:
  - <output_dir>/predictions_input.csv

CSV format: must have text content (column "text", "content", or "prompt") and
label (column "label" or "is_high_risk"). Label: 1/Yes/true = high_risk, 0/No/false = low_risk.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Project root = project/ (parent of src/)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails import (
    describe_guardrail,
    get_predictions,
    load_guardrails_from_module,
    load_evaluation_data,
    write_predictions_csv,
)
from src.guardrails.base import EvaluationType

LOGGER = logging.getLogger(__name__)
TIME_LIMIT_ENV_KEYS = (
    "PREDICTION_TIME_LIMIT_SECONDS",
    "HACKATHON_PREDICTION_TIME_LIMIT_SECONDS",
)


def _configure_logging(verbose: bool) -> None:
    """Set up CLI logging for actionable debugging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_prediction_time_limit(
    cli_time_limit_seconds: float | None,
) -> float | None:
    """
    Resolve prediction time limit from CLI first, then known env vars.
    Returns None when no valid limit is provided.
    """
    if cli_time_limit_seconds is not None:
        if cli_time_limit_seconds <= 0:
            raise ValueError("--time-limit-seconds must be > 0")
        LOGGER.info("Using prediction time limit from CLI: %.2f seconds", cli_time_limit_seconds)
        return cli_time_limit_seconds

    for key in TIME_LIMIT_ENV_KEYS:
        raw = os.getenv(key)
        if not raw:
            continue
        try:
            limit = float(raw)
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid %s=%r (must be numeric seconds)", key, raw
            )
            continue
        if limit <= 0:
            LOGGER.warning(
                "Ignoring invalid %s=%r (must be > 0 seconds)", key, raw
            )
            continue
        LOGGER.info("Using prediction time limit from env %s=%.2f seconds", key, limit)
        return limit
    return None


def run_predictions(
    submission_path: Path,
    data_path: Path,
    output_dir: Path,
    time_limit_seconds: float | None = None,
) -> Dict[str, Any]:
    """
    Load guardrails from submission, run input guardrail on data CSV,
    write predictions_input.csv to output_dir.
    Returns dict with "input_predictions_path" and "total_samples".
    """
    LOGGER.info("Loading evaluation data from: %s", data_path)
    try:
        evaluation_data = load_evaluation_data(data_path)
    except ValueError as exc:
        LOGGER.error(
            "Input dataset schema not respected. Expected one content column from "
            "['text', 'Text', 'content', 'prompt'] and one label column from "
            "['label', 'Label', 'is_high_risk']. Error: %s",
            exc,
        )
        raise
    LOGGER.info("Loaded %d evaluation rows", len(evaluation_data))
    if not evaluation_data:
        raise ValueError(f"No rows loaded from {data_path}")

    LOGGER.info("Loading guardrails from submission module: %s", submission_path)
    input_guardrail, _ = load_guardrails_from_module(submission_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory ready: %s", output_dir)
    result: Dict[str, Any] = {
        "submission": str(submission_path),
        "data_path": str(data_path),
        "total_samples": len(evaluation_data),
        "output_dir": str(output_dir),
    }

    if input_guardrail is not None:
        LOGGER.info(
            "Running input guardrail predictions | input=%s",
            describe_guardrail(input_guardrail),
        )
        prediction_start = time.perf_counter()
        input_predictions = get_predictions(
            input_guardrail,
            evaluation_data,
            evaluation_type=EvaluationType.USER_INPUT,
            include_latency=True,
            content_key="content",
            label_key="label",
        )
        elapsed_seconds = time.perf_counter() - prediction_start
        rows_generated = len(input_predictions)
        rows_per_second = (rows_generated / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        LOGGER.info("Generated %d prediction rows", len(input_predictions))
        LOGGER.info(
            "Prediction runtime summary | elapsed_seconds=%.3f rows=%d rows_per_second=%.2f",
            elapsed_seconds,
            rows_generated,
            rows_per_second,
        )
        if time_limit_seconds is not None and elapsed_seconds > time_limit_seconds:
            LOGGER.error(
                "Predictions exceeded time limit | elapsed_seconds=%.3f limit_seconds=%.3f over_by_seconds=%.3f rows=%d rows_per_second=%.2f",
                elapsed_seconds,
                time_limit_seconds,
                elapsed_seconds - time_limit_seconds,
                rows_generated,
                rows_per_second,
            )
            raise TimeoutError(
                f"Predictions took longer than time limit: {elapsed_seconds:.3f}s > "
                f"{time_limit_seconds:.3f}s (rows={rows_generated}, rows_per_second={rows_per_second:.2f})"
            )
        out_path = output_dir / "predictions.csv"
        if input_predictions:
            write_predictions_csv(input_predictions, out_path)
            LOGGER.info("Wrote predictions CSV: %s", out_path)
            LOGGER.info("Prediction CSV columns: %s", list(input_predictions[0].keys()))
            result["input_predictions_path"] = str(out_path)
        else:
            LOGGER.warning(
                "No predictions were generated by input guardrail; predictions.csv was not written."
            )
    else:
        LOGGER.error(
            "No input guardrail was loaded from submission.py get_guardrails(); cannot run predictions."
        )
        LOGGER.error(
            "Guardrails returned | input=%s",
            describe_guardrail(input_guardrail),
        )
        LOGGER.error(
            "Likely causes: model artifact/path missing, model path mismatch from hackathon.json destination, "
            "please check hackathon.json and submission.py model path configuration."
        )
        raise RuntimeError(
            "Input guardrail is None; predictions were not generated because no model/guardrail was loaded."
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run guardrails on labeled CSV and write prediction CSVs."
    )
    parser.add_argument(
        "--submission",
        "-s",
        required=True,
        help="Path to submission module (defines get_guardrails()).",
    )
    parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to CSV with content and label (e.g. text,label or prompt,is_high_risk).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to write predictions.csv.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=None,
        help=(
            "Optional prediction runtime limit in seconds. "
            "If exceeded, run fails with a clear timeout error. "
            "Can also be set via PREDICTION_TIME_LIMIT_SECONDS."
        ),
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    submission_path = Path(args.submission).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    time_limit_seconds = _resolve_prediction_time_limit(args.time_limit_seconds)
    LOGGER.info(
        "Resolved paths | submission=%s data=%s output_dir=%s",
        submission_path,
        data_path,
        output_dir,
    )
    LOGGER.info("Prediction time limit | seconds=%s", time_limit_seconds if time_limit_seconds is not None else "<none>")

    if not submission_path.exists():
        print(f"Submission not found: {submission_path}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"Data CSV not found: {data_path}", file=sys.stderr)
        return 1

    try:
        result = run_predictions(
            submission_path=submission_path,
            data_path=data_path,
            output_dir=output_dir,
            time_limit_seconds=time_limit_seconds,
        )
    except Exception as e:
        LOGGER.exception("Failed to generate predictions")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Predictions written")
    print("=" * 40)
    print(f"Samples: {result['total_samples']}")
    if result.get("input_predictions_path"):
        print(f"Predictions: {result['input_predictions_path']}")
    else:
        print("Predictions: not produced (see logs for reason)")
    print("\nRun get_guardrail_metrics with --predictions-dir to compute metrics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
