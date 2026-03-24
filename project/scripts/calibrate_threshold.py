"""
project/scripts/calibrate_threshold.py
Team 027 — Mental Health Safety Sandbox Hackathon

Reconstruct the exact val split used during training (seed=42, test_fraction=0.15),
run inference through the trained MHSGuardrail, sweep thresholds from 0→1, and
recommend the optimal _THRESHOLD value for submission.py.

Two recommendations are printed:
  - Youden's J maximizer     (balances sensitivity and specificity)
  - Recall ≥ 95% (preferred) (minimizes false negatives — safety-first)

The final line tells you exactly what to paste into submission.py.

Usage (from repo root):
    python project/scripts/calibrate_threshold.py \\
        --data         datasets/train.csv \\
        --model_dir    project/models/mhs_guardrail \\
        --device       cuda \\
        [--batch_size  4] \\
        [--seed        42] \\
        [--test_fraction 0.15] \\
        [--output      calibration_results.json]

After calibration, update submission.py:
    _THRESHOLD = <recommended value>
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# ── Signal ordering must match training script exactly ────────────────────────
SIGNALS: List[str] = [
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
N_LABELS = len(SIGNALS)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 4096  # matches submission.py _MAX_LENGTH (updated from 2048)


# ──────────────────────────────────────────────────────────────────────────────
# Val split reconstruction — must match load_rows() in train_qwen_guardrail.py
# ──────────────────────────────────────────────────────────────────────────────

def _parse_signals_cell(cell: str) -> List[float]:
    """
    Parse a signals JSON cell into a float label vector aligned to SIGNALS.
    Replicates _parse_signals_cell() from train_qwen_guardrail.py exactly.
    """
    cell = cell.strip()
    if not cell or cell in ("{}", "[]", "null"):
        return []
    try:
        obj = json.loads(cell)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        return [float(int(obj.get(sig, 0))) for sig in SIGNALS]
    if isinstance(obj, list):
        if len(obj) == N_LABELS:
            return [float(v) for v in obj]
    return []


def load_val_split(
    data_path: Path,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> List[Tuple[str, List[float], int]]:
    """
    Reconstruct the val split used during training.
    Logic is identical to load_rows() in train_qwen_guardrail.py so the
    reconstructed split is the exact same rows seen during validation.

    Returns list of (text, signal_labels_9float, binary_label) triples.
    binary_label comes from the 'label' column in train.csv — the value
    written by compute_label(), which correctly handles all label exceptions
    (pair_adversarial_negative trust-label, testing/third_party_concern deferral).
    Using the label column (not any(signals > 0.5)) is essential so that
    calibration ground-truth matches the corrected label logic.
    """
    rng = random.Random(seed)

    rows: List[Tuple[str, List[float], int]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if not text:
                continue
            labels = _parse_signals_cell(row.get("signals", ""))
            if not labels:
                binary = int(row.get("label", 0))
                labels = [float(binary)] * N_LABELS
            # Binary ground truth: always from label column, never from any(signals)
            binary_label = int(row.get("label", 0))
            rows.append((text, labels, binary_label))

    LOGGER.info("CSV rows loaded: %d", len(rows))
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * test_fraction))
    val = rows[:n_val]
    LOGGER.info("Val split reconstructed: %d samples (first %d of shuffled rows)", n_val, n_val)
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Model loading — replicates MHSGuardrail.__init__() from submission.py
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path, device: str):
    """Load Qwen2.5-7B + LoRA adapters + classifier head."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok_path = model_dir / "tokenizer"
    if not tok_path.exists():
        tok_path = model_dir  # tokenizer saved at root

    LOGGER.info("Loading tokenizer from %s", tok_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_path),
        truncation_side="left",
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    LOGGER.info("Loading base model %s (dtype=%s)…", BASE_MODEL, dtype)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )

    LOGGER.info("Applying LoRA adapters from %s", model_dir)
    backbone = PeftModel.from_pretrained(base, str(model_dir))
    backbone.eval()

    hidden_size = base.config.hidden_size
    head = torch.nn.Linear(hidden_size, N_LABELS, bias=True)
    head_path = model_dir / "classifier_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"classifier_head.pt not found at {head_path}")
    head.load_state_dict(torch.load(str(head_path), map_location=device))
    head = head.to(device=device, dtype=dtype)
    head.eval()

    LOGGER.info("Model ready | device=%s hidden_size=%d", device, hidden_size)
    return tokenizer, backbone, head


# ──────────────────────────────────────────────────────────────────────────────
# Inference — replicates MHSGuardrail.evaluate() logic for batch use
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    texts: List[str],
    tokenizer,
    backbone,
    head,
    device: str,
    batch_size: int = 1,
) -> Tuple[List[float], List[List[float]]]:
    """
    Run inference on all texts.

    Returns:
        overall_scores     : list[float], max(p_1..p_9) per sample
        signal_probs_all   : list[list[float]], all 9 probabilities per sample
    """
    import torch

    overall_scores: List[float] = []
    signal_probs_all: List[List[float]] = []
    n = len(texts)
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Last token hidden state → head → sigmoid — matches submission.py
            last_hidden = outputs.hidden_states[-1][:, -1, :]   # (B, hidden)
            logits = head(last_hidden)                            # (B, 9)
            probs  = torch.sigmoid(logits).cpu().float().tolist()  # [[9], ...]

            for p_list in probs:
                overall_scores.append(max(p_list))
                signal_probs_all.append(p_list)

            done = min(i + batch_size, n)
            elapsed = time.perf_counter() - t_start
            spd = elapsed / done
            eta = spd * (n - done)
            LOGGER.info(
                "Inference: %d/%d  |  %.2f s/sample  |  ETA %.0f s",
                done, n, spd, eta,
            )

    return overall_scores, signal_probs_all


# ──────────────────────────────────────────────────────────────────────────────
# Metrics and threshold sweep
# ──────────────────────────────────────────────────────────────────────────────

def _metrics_at(
    scores: List[float],
    y_true: List[bool],
    threshold: float,
) -> Dict:
    tp = fp = fn = tn = 0
    for s, t in zip(scores, y_true):
        pred = s >= threshold
        if pred and t:      tp += 1
        elif pred:          fp += 1
        elif t:             fn += 1
        else:               tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    youden_j  = recall - fpr  # TPR - FPR

    return {
        "threshold": round(threshold, 6),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "youden_j":  round(youden_j, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sweep_thresholds(
    scores: List[float],
    y_true: List[bool],
    n_steps: int = 400,
) -> List[Dict]:
    """Sweep n_steps+1 thresholds across the observed score range."""
    lo, hi = min(scores), max(scores)
    # Always include 0.5 (current default) explicitly
    step_values = [lo + (hi - lo) * i / n_steps for i in range(n_steps + 1)]
    if 0.5 not in step_values:
        step_values.append(0.5)
    step_values.sort()
    return [_metrics_at(scores, y_true, t) for t in step_values]


def _best_youden(sweep: List[Dict]) -> Dict:
    return max(sweep, key=lambda x: x["youden_j"])


def _recall_constrained(sweep: List[Dict], min_recall: float) -> Optional[Dict]:
    """Highest-precision row where recall >= min_recall."""
    candidates = [r for r in sweep if r["recall"] >= min_recall]
    return max(candidates, key=lambda x: x["precision"]) if candidates else None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Logistic regression aggregation head
#
# Replaces max(p_1…p_9) with a learned weighted sum.
# Trained on val split signal probability vectors + corrected binary labels.
#
# Why val split and not training data:
#   Qwen's outputs on training data are overconfident (optimised on them in
#   phase 1).  Val split probabilities are "honest" — Qwen never saw these
#   rows — so the LR learns weights that generalise to deployment.
#
# Saved to:
#   <model_dir>/aggregation_head.json  — weights + bias (load in submission.py)
# ──────────────────────────────────────────────────────────────────────────────

def fit_aggregation_head(
    signal_probs: List[List[float]],
    y_true: List[bool],
    model_dir: Path,
) -> Dict:
    """
    Fit a logistic regression on the 9 val-set signal probability vectors.

    Returns a dict with:
        weights     : list[float] of length 9, one weight per signal
        bias        : float
        agg_scores  : list[float], P(high_risk) from LR for each val sample
        val_metrics : dict, precision / recall / F1 at threshold=0.5
        signal_weights : dict, signal_name → weight (for human inspection)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except ImportError:
        LOGGER.error(
            "scikit-learn not installed. Run: pip install scikit-learn\n"
            "Skipping aggregation head fitting."
        )
        return {}

    X = np.array(signal_probs, dtype=np.float32)   # (N, 9)
    y = np.array([int(t) for t in y_true], dtype=np.int32)  # (N,)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    LOGGER.info(
        "Fitting LogisticRegression(C=0.1) on %d val samples "
        "(%d high_risk, %d low_risk)…",
        len(y), n_pos, n_neg,
    )

    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf.fit(X, y)

    weights: List[float] = clf.coef_[0].tolist()
    bias: float = float(clf.intercept_[0])

    # LR probability scores on val set (used for comparison against max-sweep)
    agg_scores: List[float] = clf.predict_proba(X)[:, 1].tolist()

    # Metrics at threshold=0.5 (natural operating point of LR output)
    val_metrics = _metrics_at(agg_scores, list(y_true), threshold=0.5)

    # Per-signal weight table (most interpretable output)
    signal_weights = {s: round(w, 4) for s, w in zip(SIGNALS, weights)}

    # Save to model_dir/aggregation_head.json
    head_path = model_dir / "aggregation_head.json"
    payload = {"weights": [round(w, 6) for w in weights], "bias": round(bias, 6)}
    with open(head_path, "w") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Aggregation head saved to %s", head_path)

    return {
        "weights":        weights,
        "bias":           bias,
        "agg_scores":     agg_scores,
        "val_metrics":    val_metrics,
        "signal_weights": signal_weights,
        "head_path":      str(head_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Calibrate guardrail threshold on the training val split"
    )
    p.add_argument("--data",          default="datasets/train.csv",
                   help="Path to train.csv used for training")
    p.add_argument("--model_dir",     default="project/models/mhs_guardrail",
                   help="Path to trained model directory")
    p.add_argument("--device",        default="cpu",
                   help="Inference device: cpu | cuda | cuda:0 etc.")
    p.add_argument("--batch_size",    type=int, default=1,
                   help="Inference batch size (increase if GPU has headroom)")
    p.add_argument("--seed",          type=int, default=42,
                   help="Must match --seed used during training")
    p.add_argument("--test_fraction", type=float, default=0.15,
                   help="Must match --test_fraction used during training")
    p.add_argument("--output",        default="calibration_results.json",
                   help="Path to write full results JSON")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        LOGGER.error("Model directory not found: %s", model_dir)
        LOGGER.error("Run scripts/run_train.sh first.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        LOGGER.error("Data file not found: %s", data_path)
        sys.exit(1)

    # ── 1. Reconstruct val split ──────────────────────────────────────────────
    LOGGER.info("Reconstructing val split (seed=%d, test_fraction=%.2f)…",
                args.seed, args.test_fraction)
    val_rows   = load_val_split(data_path, args.test_fraction, args.seed)
    val_texts  = [r[0] for r in val_rows]
    val_labels = [r[1] for r in val_rows]  # list of 9-float signal vectors

    # Binary ground truth: from label column in train.csv (written by compute_label()).
    # This correctly handles all label exceptions — notably pair_adversarial_negative
    # rows stay label=0 even when signals are 1.  Using any(signal > 0.5) here would
    # re-introduce the contamination that compute_label() was fixed to avoid.
    y_true = [bool(r[2]) for r in val_rows]
    n_pos  = sum(y_true)
    n_neg  = len(y_true) - n_pos
    LOGGER.info("Val split: %d total | %d high_risk | %d low_risk",
                len(y_true), n_pos, n_neg)

    if n_pos == 0 or n_neg == 0:
        LOGGER.error(
            "Val split has no %s samples — threshold calibration is meaningless.",
            "high_risk" if n_pos == 0 else "low_risk",
        )
        sys.exit(1)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    tokenizer, backbone, head = load_model(model_dir, args.device)

    # ── 3. Run inference ──────────────────────────────────────────────────────
    LOGGER.info("Running inference on %d val samples…", len(val_texts))
    overall_scores, signal_probs_all = run_inference(
        val_texts, tokenizer, backbone, head, args.device, args.batch_size
    )

    # Baseline at 0.5
    baseline = _metrics_at(overall_scores, y_true, 0.5)
    LOGGER.info(
        "Baseline @ 0.50 | P=%.4f R=%.4f F1=%.4f J=%.4f",
        baseline["precision"], baseline["recall"],
        baseline["f1"], baseline["youden_j"],
    )

    # ── 4. Sweep ──────────────────────────────────────────────────────────────
    sweep = sweep_thresholds(overall_scores, y_true, n_steps=400)

    opt = _best_youden(sweep)
    t95 = _recall_constrained(sweep, min_recall=0.95)
    t90 = _recall_constrained(sweep, min_recall=0.90)
    t80 = _recall_constrained(sweep, min_recall=0.80)

    # Safety classifier recommendation: recall ≥ 95% if achievable, else Youden
    if t95:
        recommended = t95
        rec_reason  = "recall ≥ 95%  (minimises false negatives for high-risk users)"
    else:
        recommended = opt
        rec_reason  = "Youden J max  (recall ≥ 95% not achievable on this val set)"

    # ── 5. Print report ───────────────────────────────────────────────────────
    w = 74
    print()
    print("=" * w)
    print("  GUARDRAIL THRESHOLD CALIBRATION — team_027")
    print(f"  Val split : {len(y_true)} samples  |  {n_pos} high_risk  |  {n_neg} low_risk")
    print(f"  Model     : {model_dir}")
    print("=" * w)
    print()

    # Threshold table (every 20th step + key rows)
    key_t = {baseline["threshold"], opt["threshold"]}
    for r in [t95, t90, t80]:
        if r:
            key_t.add(r["threshold"])

    hdr = f"{'Threshold':>10}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Youden J':>9}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}"
    print(hdr)
    print("-" * len(hdr))

    printed = set()
    for i, row in enumerate(sweep):
        is_key = row["threshold"] in key_t
        if i % 20 == 0 or is_key:
            t = row["threshold"]
            if t in printed:
                continue
            printed.add(t)
            tag = ""
            if t == opt["threshold"]:
                tag = "  ← Youden max"
            if t95 and t == t95["threshold"]:
                tag = "  ← recall ≥ 95%"
            if t90 and t == t90["threshold"] and not tag:
                tag = "  ← recall ≥ 90%"
            if t == baseline["threshold"] and not tag:
                tag = "  ← current default"
            print(
                f"{row['threshold']:>10.4f}  {row['precision']:>9.4f}  "
                f"{row['recall']:>7.4f}  {row['f1']:>7.4f}  {row['youden_j']:>9.4f}  "
                f"{row['tp']:>4}  {row['fp']:>4}  {row['fn']:>4}  {row['tn']:>4}{tag}"
            )

    print()
    print("─" * w)
    print("  RECOMMENDATIONS")
    print("─" * w)
    print()

    def _fmt(label: str, r: Optional[Dict]) -> None:
        if r is None:
            print(f"  {label}: NOT ACHIEVABLE on this val set")
            return
        print(f"  {label}: threshold = {r['threshold']:.4f}")
        print(f"    precision={r['precision']:.4f}  recall={r['recall']:.4f}  "
              f"F1={r['f1']:.4f}  Youden={r['youden_j']:.4f}")
        print(f"    TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")
        print()

    _fmt("Youden J maximizer     ", opt)
    _fmt("Best precision @ R≥95%", t95)
    _fmt("Best precision @ R≥90%", t90)
    _fmt("Best precision @ R≥80%", t80)

    print(f"  RECOMMENDED ({rec_reason}):")
    print(f"    threshold = {recommended['threshold']:.4f}")
    print()
    print("  ─── Copy this line into project/src/submission/submission.py ───")
    print()
    print(f"  _THRESHOLD = {recommended['threshold']:.4f}   "
          f"# calibrated on val split (seed={args.seed}, n={len(y_true)})")
    print()
    print("=" * w)
    print()

    # ── 6. Fit logistic regression aggregation head (Phase 2) ─────────────────
    LOGGER.info("Fitting logistic regression aggregation head (Phase 2)…")
    agg = fit_aggregation_head(signal_probs_all, y_true, model_dir)

    if agg:
        am = agg["val_metrics"]
        print("─" * w)
        print("  AGGREGATION HEAD  (logistic regression on 9 signal probabilities)")
        print("─" * w)
        print()
        print("  Learned signal weights (higher = more predictive of high_risk):")
        sw = agg["signal_weights"]
        for sig in SIGNALS:
            bar = "█" * max(0, int((sw[sig] + 3) * 4))   # rough visual bar
            print(f"    {sig:<25} {sw[sig]:+.4f}  {bar}")
        print()
        print(f"  Bias: {agg['bias']:+.4f}")
        print()
        print(f"  Val metrics at threshold=0.5  (LR natural operating point):")
        print(f"    Precision={am['precision']:.4f}  Recall={am['recall']:.4f}  "
              f"F1={am['f1']:.4f}  Youden={am['youden_j']:.4f}")
        print(f"    TP={am['tp']}  FP={am['fp']}  FN={am['fn']}  TN={am['tn']}")
        print()

        # Compare against the max(p) recommended threshold
        rmax = _metrics_at(overall_scores, y_true, recommended["threshold"])
        print(f"  Compare — max(p) @ recommended threshold={recommended['threshold']:.4f}:")
        print(f"    Precision={rmax['precision']:.4f}  Recall={rmax['recall']:.4f}  "
              f"F1={rmax['f1']:.4f}  Youden={rmax['youden_j']:.4f}")
        print(f"    TP={rmax['tp']}  FP={rmax['fp']}  FN={rmax['fn']}  TN={rmax['tn']}")
        print()

        print("  ─── Add to submission.py ───────────────────────────────────────")
        print()
        print("  import json, math")
        print("  _AGG_HEAD = json.load(open(model_path / 'aggregation_head.json'))")
        print("  _AGG_W    = _AGG_HEAD['weights']   # list of 9 floats")
        print("  _AGG_B    = _AGG_HEAD['bias']       # float")
        print()
        print("  # Replace:  score = max(probs)")
        print("  # With:")
        print("  z     = sum(w * p for w, p in zip(_AGG_W, probs)) + _AGG_B")
        print("  score = 1.0 / (1.0 + math.exp(-z))  # sigmoid → P(high_risk)")
        print("  # Threshold stays 0.5 (natural LR operating point)")
        print()
        print("=" * w)
        print()

    # ── 7. Save JSON ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "val_size":              len(y_true),
        "n_high_risk":           n_pos,
        "n_low_risk":            n_neg,
        "recommended_threshold": recommended["threshold"],
        "recommended_reason":    rec_reason,
        "optimal_youden":        opt,
        "recall_95_threshold":   t95,
        "recall_90_threshold":   t90,
        "recall_80_threshold":   t80,
        "baseline_at_0_5":       baseline,
        "aggregation_head": {
            "signal_weights": agg.get("signal_weights") if agg else None,
            "bias":           round(agg["bias"], 6) if agg else None,
            "val_metrics":    agg.get("val_metrics") if agg else None,
            "head_path":      agg.get("head_path") if agg else None,
        },
        "sweep":                 sweep,
        "per_sample": [
            {
                "score":        round(s, 6),
                "agg_score":    round(a, 6) if agg else None,
                "signal_probs": [round(v, 6) for v in sp],
                "y_true":       yt,
            }
            for s, sp, yt, a in zip(
                overall_scores, signal_probs_all, y_true,
                agg.get("agg_scores", [None] * len(y_true)) if agg else [None] * len(y_true),
            )
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Full results saved to %s", out_path)


if __name__ == "__main__":
    main()
