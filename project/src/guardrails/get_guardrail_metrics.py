#!/usr/bin/env python3
"""Compute precision, recall, F1 and latency from prediction CSV.

Reads predictions produced by get_predictions and computes metrics.
Supports single or stacked guardrails (combined_pred = high_risk if any
guardrail said high_risk).

Usage:
    cd project && PYTHONPATH=. python -m src.guardrails.get_guardrail_metrics \\
        --predictions-dir results/ \\
        [--output results/metrics.json]

    Or specify file explicitly:
    python -m src.guardrails.get_guardrail_metrics \\
        --predictions results/predictions_input.csv \\
        --output results/metrics.json

When --output is set, writes:
  - <output>.json (full metrics JSON)
  - <output_dir>/metrics.csv (precision, recall, F1, latency)

Prediction CSV format: must have columns label_high_risk (or label) and combined_pred;
optional latency_ms. Use get_predictions to generate from a submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root = project/ (parent of src/)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails import (
    GuardrailMetricsResult,
    compute_metrics_from_predictions,
)

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Set up CLI logging for actionable debugging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_predictions_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load prediction CSV; preserve numeric/string values (metrics layer normalizes)."""
    LOGGER.info("Reading predictions CSV: %s", csv_path)
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            LOGGER.warning("Predictions CSV has no header/fieldnames: %s", csv_path)
            return rows
        LOGGER.debug("Predictions CSV columns: %s", reader.fieldnames)
        for row in reader:
            # Keep values as-is; compute_metrics_from_predictions handles 0/1 and labels
            rows.append(dict(row))
    LOGGER.info("Loaded %d prediction rows", len(rows))
    return rows


def _metrics_to_dict(m: GuardrailMetricsResult) -> Dict[str, Any]:
    """Serialize GuardrailMetricsResult for JSON output."""
    d: Dict[str, Any] = {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "support_high_risk": m.support_high_risk,
        "support_low_risk": m.support_low_risk,
        "total_samples": m.total_samples,
        "guardrail_names": m.guardrail_names,
    }
    if m.latency_ms_mean is not None:
        d["latency_ms_mean"] = m.latency_ms_mean
    if m.latency_ms_total is not None:
        d["latency_ms_total"] = m.latency_ms_total
    if m.latency_ms_per_sample is not None:
        d["latency_ms_per_sample_count"] = len(m.latency_ms_per_sample)
    return d


def _write_metrics_csv(
    rows: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    """Write one row with precision, recall, F1, latency."""
    if not rows:
        LOGGER.warning("No metric rows to write to CSV: %s", csv_path)
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote metrics CSV: %s", csv_path)


def run_metrics(
    predictions_path: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load predictions CSV, compute metrics, optionally write JSON and metrics CSV.

    Either pass predictions_dir (uses predictions.csv) or predictions_path.
    """
    if predictions_dir is not None:
        predictions_dir = predictions_dir.resolve()
        LOGGER.info("Using predictions directory: %s", predictions_dir)
        if predictions_path is None:
            p = predictions_dir / "predictions.csv"
            if p.exists():
                predictions_path = p
                LOGGER.info("Found predictions file in directory: %s", predictions_path)
            else:
                csv_candidates = sorted(
                    [str(fp.name) for fp in predictions_dir.glob("*.csv")]
                )
                LOGGER.error(
                    "Expected predictions file not found at %s. Available CSV files in directory: %s",
                    p,
                    csv_candidates or "<none>",
                )
                raise FileNotFoundError(
                    f"Predictions file was not produced. Expected {p}. "
                    f"Found CSVs: {csv_candidates or 'none'}. "
                    "Run get_predictions first and ensure it writes predictions.csv."
                )

    if predictions_path is None:
        raise ValueError("Provide --predictions-dir or --predictions")
    predictions_path = predictions_path.resolve()
    LOGGER.info("Using predictions path: %s", predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    results: Dict[str, Any] = {
        "total_samples": None,
        "guardrail": None,
    }
    metrics_csv_rows: List[Dict[str, Any]] = []
    out_dir = output_path.parent if output_path else None

    predictions = _load_predictions_csv(predictions_path)
    if predictions:
        metrics = compute_metrics_from_predictions(
            predictions,
            combined_pred_key="combined_pred",
            label_key="label_high_risk",
            fallback_label_key="label",
            latency_key="latency_ms",
        )
        LOGGER.info(
            "Computed metrics on %d samples: precision=%.4f recall=%.4f f1=%.4f",
            metrics.total_samples,
            metrics.precision,
            metrics.recall,
            metrics.f1,
        )
        results["guardrail"] = _metrics_to_dict(metrics)
        results["total_samples"] = metrics.total_samples
        metrics_csv_rows.append({
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "latency_ms_mean": metrics.latency_ms_mean if metrics.latency_ms_mean is not None else "",
            "latency_ms_total": metrics.latency_ms_total if metrics.latency_ms_total is not None else "",
            "support_high_risk": metrics.support_high_risk,
            "support_low_risk": metrics.support_low_risk,
            "total_samples": metrics.total_samples,
            "guardrail_names": "|".join(metrics.guardrail_names),
        })
    else:
        LOGGER.warning(
            "Predictions CSV exists but has no data rows. Metrics remain empty."
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        LOGGER.info("Wrote metrics JSON: %s", output_path)
        if out_dir is not None and metrics_csv_rows:
            _write_metrics_csv(metrics_csv_rows, out_dir / "metrics.csv")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute precision, recall, F1 and latency from prediction CSVs (from get_predictions)."
    )
    parser.add_argument(
        "--predictions-dir",
        "-p",
        default=None,
        help="Directory containing predictions.csv.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to predictions CSV (overrides predictions-dir).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to write JSON metrics (and metrics.csv in same dir).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    predictions_dir = Path(args.predictions_dir).expanduser().resolve() if args.predictions_dir else None
    predictions_path = Path(args.predictions).expanduser().resolve() if args.predictions else None
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    LOGGER.info(
        "Resolved args | predictions_dir=%s predictions=%s output=%s",
        predictions_dir,
        predictions_path,
        output_path,
    )

    if predictions_dir is not None and not predictions_dir.exists():
        print(f"Predictions dir not found: {predictions_dir}", file=sys.stderr)
        return 1

    try:
        results = run_metrics(
            predictions_path=predictions_path,
            predictions_dir=predictions_dir,
            output_path=output_path,
        )
    except Exception as e:
        LOGGER.exception("Failed to compute guardrail metrics")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print summary
    print("Guardrail metrics (positive class = high_risk)")
    print("=" * 50)
    if results.get("guardrail"):
        m = results["guardrail"]
        if "error" in m:
            print("Guardrail:", m["error"])
        else:
            print(f"  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")
            if "latency_ms_mean" in m:
                print(f"  Latency: mean={m['latency_ms_mean']:.2f} ms  total={m['latency_ms_total']:.2f} ms")
            print(f"  Samples: {m['total_samples']} (high_risk={m['support_high_risk']}, low_risk={m['support_low_risk']})")
            if m.get("guardrail_names"):
                print(f"  Guardrails: {m['guardrail_names']}")

    if output_path:
        out_dir = output_path.parent
        print(f"Metrics JSON: {output_path}")
        if (out_dir / "metrics.csv").exists():
            print(f"Metrics CSV: {out_dir / 'metrics.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
