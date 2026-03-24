"""
scripts/train_qwen_guardrail.py
Train Qwen2.5-7B-Instruct + LoRA as a 9-head multi-label safety classifier.

Architecture
────────────
  Backbone  : Qwen/Qwen2.5-7B-Instruct (frozen except LoRA delta weights)
  Adapter   : PEFT LoRA  r=16, α=32, dropout=0.05
              target_modules = [q_proj, k_proj, v_proj, o_proj]
  Head      : Linear(hidden_size, 9, bias=True)  — one sigmoid per signal
  Loss      : BCEWithLogitsLoss (independent multi-label objective)
  Dtype     : bfloat16  (A6000 48 GB, no quantization needed)

Input CSV (train.csv)
──────────────────────
  text   : full conversation string (may be truncated left to max_length tokens)
  label  : not used directly — signals JSON column drives multi-label targets
  signals: JSON dict  {"burden_language": 1, "hopelessness": 0, ...}

  If 'signals' column is absent, falls back to binary 'label' column:
  label=1 → all signals ON, label=0 → all signals OFF.

Outputs saved to --output_dir (default: project/models/mhs_guardrail/)
  ├── adapter_config.json          (LoRA config — written by PEFT)
  ├── adapter_model.safetensors    (LoRA delta weights — written by PEFT)
  ├── classifier_head.pt           (linear head state_dict — torch.save)
  ├── tokenizer/                   (tokenizer files — saved by HF tokenizer)
  ├── checkpoint_last/             (end-of-epoch checkpoint, overwritten each epoch)
  │   ├── peft_state.pt            model.state_dict()
  │   ├── classifier_head.pt
  │   ├── optimizer.pt
  │   ├── scheduler.pt
  │   └── training_state.json      {"completed_epoch": N, "best_macro_f1": X}
  └── checkpoint_best/             (copy of checkpoint_last at the epoch with best val F1)

Usage
─────
  python project/scripts/train_qwen_guardrail.py \\
    --data         datasets/train.csv \\
    --output_dir   project/models/mhs_guardrail \\
    [--base_model  Qwen/Qwen2.5-7B-Instruct] \\
    [--lora_r 16] [--lora_alpha 32] [--lora_dropout 0.05] \\
    [--lora_targets q_proj,k_proj,v_proj,o_proj] \\
    [--epochs 3] [--batch_size 4] [--grad_accum 4] \\
    [--lr 2e-4] [--warmup_ratio 0.05] \\
    [--max_length 2048] [--threshold 0.5] [--dtype bfloat16] \\
    [--test_fraction 0.15] [--seed 42] [--resume]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional


from config import (
    SIGNALS,
    CLASSIFIER_BASE_MODEL,
    CLASSIFIER_DTYPE,
    CLASSIFIER_OUTPUT_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_GRAD_ACCUM,
    CLASSIFIER_LR,
    CLASSIFIER_WARMUP_RATIO,
    CLASSIFIER_MAX_LENGTH,
    CLASSIFIER_THRESHOLD,
    N_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _parse_signals_cell(cell: str) -> list[float]:
    """
    Convert a signals JSON cell into a float label vector aligned to SIGNALS.

    Accepts two formats:
        dict  {"burden_language": 1, "hopelessness": 0, ...}
        list  [1, 0, 0, 1, ...]   (already aligned)
    Returns an empty list on parse failure.
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


def load_rows(path: Path, test_fraction: float, seed: int):
    """
    Load train.csv → (train_texts, train_labels, val_texts, val_labels).

    Labels are list[float] of length N_LABELS (9).
    Falls back to binary label column when signals column is missing / empty.
    """
    import random
    rng = random.Random(seed)

    rows: list[tuple[str, list[float]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if not text:
                continue

            labels = _parse_signals_cell(row.get("signals", ""))
            if not labels:
                # Fallback: binary label → replicate across all signals
                binary = int(row.get("label", 0))
                labels = [float(binary)] * N_LABELS

            rows.append((text, labels))

    LOGGER.info("Loaded %d rows from %s", len(rows), path)

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * test_fraction))
    val   = rows[:n_val]
    train = rows[n_val:]
    LOGGER.info("Train: %d  |  Val: %d", len(train), len(val))

    train_texts  = [r[0] for r in train]
    train_labels = [r[1] for r in train]
    val_texts    = [r[0] for r in val]
    val_labels   = [r[1] for r in val]
    return train_texts, train_labels, val_texts, val_labels


# ──────────────────────────────────────────────────────────────────────────────
# Collation
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloader(texts, labels, tokenizer, batch_size: int, max_length: int, shuffle: bool):
    """Return a simple list-of-batches (no torch.utils.data dependency at import time)."""
    import torch

    batches = []
    indices = list(range(len(texts)))
    if shuffle:
        import random
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_texts  = [texts[i] for i in batch_idx]
        batch_labels = [labels[i] for i in batch_idx]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,          # pad to longest in batch (dynamic padding)
            padding_side="left",
        )
        label_tensor = torch.tensor(batch_labels, dtype=torch.float32)
        batches.append({
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         label_tensor,
        })
    return batches


# ──────────────────────────────────────────────────────────────────────────────
# Model initialisation
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args):
    """Load base model, attach LoRA adapters, and build classifier head."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    LOGGER.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        truncation_side="left",   # keep most recent turns
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading %s (dtype=%s, attn=sdpa) …", args.base_model, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False   # required for gradient checkpointing

    lora_targets = [t.strip() for t in args.lora_targets.split(",")]
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    LOGGER.info(
        "Wrapping with LoRA  r=%d  α=%d  dropout=%.2f  targets=%s",
        args.lora_r, args.lora_alpha, args.lora_dropout, lora_targets,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing: trades compute for ~30-40% VRAM reduction during
    # backprop.  Must be enabled AFTER get_peft_model(); use_cache=False already set.
    # enable_input_require_grads() is required so that gradient checkpointing can
    # back-propagate through frozen base layers into LoRA branches (PEFT known issue).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    LOGGER.info("Gradient checkpointing enabled (with input_require_grads).")

    hidden_size = base.config.hidden_size
    head = torch.nn.Linear(hidden_size, N_LABELS, bias=True)
    head = head.to(device=args.device, dtype=dtype)

    return model, head, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, head, val_batches, device, threshold: float):
    """Return dict of val metrics: loss, per-signal F1, macro-F1."""
    import torch

    model.eval()
    head.eval()

    total_loss   = 0.0
    all_preds: list[list[int]] = []
    all_golds: list[list[int]] = []
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_batches:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_labels    = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = head(last_hidden)
            loss   = criterion(logits, gold_labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int().cpu().tolist()
            golds = gold_labels.int().cpu().tolist()
            all_preds.extend(preds)
            all_golds.extend(golds)

    # Per-signal F1
    tp = [0] * N_LABELS
    fp = [0] * N_LABELS
    fn = [0] * N_LABELS
    for pred_row, gold_row in zip(all_preds, all_golds):
        for i, (p, g) in enumerate(zip(pred_row, gold_row)):
            if p == 1 and g == 1:
                tp[i] += 1
            elif p == 1 and g == 0:
                fp[i] += 1
            elif p == 0 and g == 1:
                fn[i] += 1

    f1_scores = []
    for i in range(N_LABELS):
        denom = 2 * tp[i] + fp[i] + fn[i]
        f1 = (2 * tp[i] / denom) if denom > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / N_LABELS
    avg_loss = total_loss / max(1, len(val_batches))

    return {
        "val_loss":   round(avg_loss, 4),
        "macro_f1":   round(macro_f1, 4),
        "signal_f1":  {sig: round(f, 4) for sig, f in zip(SIGNALS, f1_scores)},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    import torch
    from transformers import get_linear_schedule_with_warmup

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        LOGGER.error("Training data not found: %s", data_path)
        sys.exit(1)

    train_texts, train_labels, val_texts, val_labels = load_rows(
        data_path, args.test_fraction, args.seed
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, head, tokenizer = build_model(args)
    device = args.device

    # ── Dataloaders ───────────────────────────────────────────────────────────
    LOGGER.info("Tokenising train set …")
    train_batches = make_dataloader(
        train_texts, train_labels, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length, shuffle=True,
    )
    LOGGER.info("Tokenising val set …")
    val_batches = make_dataloader(
        val_texts, val_labels, tokenizer,
        batch_size=args.batch_size * 2, max_length=args.max_length, shuffle=False,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable_params = list(model.parameters()) + list(head.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps   = math.ceil(len(train_batches) / args.grad_accum) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    LOGGER.info(
        "Training: epochs=%d  batches/epoch=%d  grad_accum=%d  "
        "total_steps=%d  warmup=%d  lr=%.0e",
        args.epochs, len(train_batches), args.grad_accum,
        total_steps, warmup_steps, args.lr,
    )

    output_dir = Path(args.output_dir)
    ckpt_last = output_dir / "checkpoint_last"
    ckpt_best = output_dir / "checkpoint_best"

    # ── Resume from last checkpoint (if requested) ────────────────────────────
    start_epoch   = 1
    best_macro_f1 = -1.0

    if args.resume:
        state_file = ckpt_last / "training_state.json"
        if state_file.exists():
            with open(state_file) as _f:
                ckpt_state = json.load(_f)
            completed = int(ckpt_state.get("completed_epoch", 0))
            if completed >= args.epochs:
                LOGGER.warning(
                    "Checkpoint shows all %d epochs completed — nothing to resume.",
                    args.epochs,
                )
            else:
                LOGGER.info(
                    "Resuming from checkpoint: completed_epoch=%d  best_macro_f1=%.4f",
                    completed, ckpt_state.get("best_macro_f1", -1.0),
                )
                # Rebuild base model, re-wrap with LoRA, then load saved weights
                peft_state  = torch.load(ckpt_last / "peft_state.pt",  map_location="cpu")
                head_state  = torch.load(ckpt_last / "classifier_head.pt", map_location="cpu")
                model.load_state_dict(peft_state)
                head.load_state_dict(head_state)
                opt_ckpt = ckpt_last / "optimizer.pt"
                sch_ckpt = ckpt_last / "scheduler.pt"
                if opt_ckpt.exists():
                    optimizer.load_state_dict(torch.load(opt_ckpt, map_location="cpu"))
                if sch_ckpt.exists():
                    scheduler.load_state_dict(torch.load(sch_ckpt, map_location="cpu"))
                start_epoch   = completed + 1
                best_macro_f1 = float(ckpt_state.get("best_macro_f1", -1.0))
                LOGGER.info("Resuming from epoch %d.", start_epoch)
        else:
            LOGGER.warning(
                "--resume flag set but no checkpoint found at %s; starting from scratch.",
                ckpt_last,
            )

    def _save_checkpoint(directory: Path, epoch_num: int) -> None:
        """Persist model state, head, optimiser, scheduler, and metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   directory / "peft_state.pt")
        torch.save(head.state_dict(),      directory / "classifier_head.pt")
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(scheduler.state_dict(), directory / "scheduler.pt")
        with open(directory / "training_state.json", "w") as _f:
            json.dump({
                "completed_epoch": epoch_num,
                "best_macro_f1":   best_macro_f1,
                "seed":            args.seed,
            }, _f, indent=2)

    updates_per_epoch = math.ceil(len(train_batches) / args.grad_accum)
    epoch_times: list[float] = []   # track per-epoch wall time for ETA

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        head.train()
        optimizer.zero_grad()

        epoch_loss   = 0.0
        step_count   = 0
        update_count = 0
        t0 = time.time()

        # Re-shuffle each epoch
        import random
        random.seed(args.seed + epoch)
        random.shuffle(train_batches)

        for step, batch in enumerate(train_batches):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_labels    = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, hidden)
            logits      = head(last_hidden)                      # (B, 9)
            loss        = criterion(logits, gold_labels)
            loss        = loss / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            step_count += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_batches):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_count += 1

                # Log every update for the first 20 (reveals true speed fast),
                # then every 10 updates — always show s/it and ETA.
                log_now = (update_count <= 20) or (update_count % 10 == 0)
                if log_now:
                    elapsed_so_far    = time.time() - t0
                    secs_per_update   = elapsed_so_far / update_count
                    updates_remaining = updates_per_epoch - update_count
                    eta_epoch_secs    = secs_per_update * updates_remaining
                    epochs_done_so_far = (epoch - start_epoch)
                    updates_done_total = epochs_done_so_far * updates_per_epoch + update_count
                    updates_total_run  = (args.epochs - start_epoch + 1) * updates_per_epoch
                    updates_left_total = updates_total_run - updates_done_total
                    eta_total_secs     = secs_per_update * updates_left_total
                    LOGGER.info(
                        "  epoch %d  update %d/%d  loss=%.4f  lr=%.2e  "
                        "%.1fs/it  eta_epoch=%dm%02ds  eta_total=%dh%02dm",
                        epoch, update_count, updates_per_epoch,
                        epoch_loss / step_count,
                        scheduler.get_last_lr()[0],
                        secs_per_update,
                        int(eta_epoch_secs) // 60, int(eta_epoch_secs) % 60,
                        int(eta_total_secs) // 3600,
                        (int(eta_total_secs) % 3600) // 60,
                    )

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_train_loss = epoch_loss / max(1, step_count)

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = evaluate(model, head, val_batches, device, args.threshold)

        # ETA across remaining epochs (use rolling average of completed epochs)
        epochs_remaining   = args.epochs - epoch
        avg_epoch_time     = sum(epoch_times) / len(epoch_times)
        eta_total_secs     = avg_epoch_time * epochs_remaining
        LOGGER.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  macro_f1=%.4f  "
            "elapsed=%.0fs  eta_remaining=%.0fs (~%.1fh)",
            epoch, args.epochs,
            avg_train_loss, val_metrics["val_loss"], val_metrics["macro_f1"],
            elapsed, eta_total_secs, eta_total_secs / 3600,
        )
        LOGGER.info("  Per-signal F1:")
        for sig, f1 in val_metrics["signal_f1"].items():
            LOGGER.info("    %-30s %.4f", sig, f1)

        # ── Track best; update best_macro_f1 BEFORE saving ckpt_last so that
        #    training_state.json always reflects the true best seen so far ─────
        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            _save_checkpoint(ckpt_best, epoch)
            LOGGER.info(
                "  ✓ New best macro_f1=%.4f — checkpoint_best saved → %s",
                best_macro_f1, ckpt_best,
            )

        # ── Save end-of-epoch checkpoint (checkpoint_last) ────────────────────
        # Saved AFTER best_macro_f1 update so training_state.json is accurate
        _save_checkpoint(ckpt_last, epoch)
        LOGGER.info("  Checkpoint saved → %s", ckpt_last)

    # ── Restore best weights for final model export ───────────────────────────
    best_peft  = ckpt_best / "peft_state.pt"
    best_hd    = ckpt_best / "classifier_head.pt"
    if best_peft.exists() and best_hd.exists():
        model.load_state_dict(torch.load(best_peft, map_location=device))
        head.load_state_dict(torch.load(best_hd,   map_location=device))
        LOGGER.info("Restored best checkpoint (macro_f1=%.4f) for final export.", best_macro_f1)
    else:
        LOGGER.warning("Best checkpoint files not found; exporting current model state.")

    # ── Save final model outputs ──────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving LoRA adapters to %s …", output_dir)
    model.save_pretrained(str(output_dir))   # writes adapter_config.json + adapter_model.safetensors

    head_path = output_dir / "classifier_head.pt"
    LOGGER.info("Saving classifier head to %s …", head_path)
    torch.save(head.state_dict(), str(head_path))

    tok_path = output_dir / "tokenizer"
    LOGGER.info("Saving tokenizer to %s …", tok_path)
    tok_path.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(tok_path))

    LOGGER.info(
        "\n"
        "══════════════════════════════════════════════════\n"
        " Training complete\n"
        " Best val macro-F1 : %.4f\n"
        " Output dir        : %s\n"
        "   adapter_config.json\n"
        "   adapter_model.safetensors\n"
        "   classifier_head.pt\n"
        "   tokenizer/\n"
        "══════════════════════════════════════════════════",
        best_macro_f1, output_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train Qwen2.5-7B-Instruct + LoRA as a 9-head multi-label safety classifier."
    )
    # Data
    p.add_argument("--data",          default="datasets/train.csv",
                   help="Path to train.csv (text + signals columns)")
    p.add_argument("--test_fraction", type=float, default=0.15,
                   help="Fraction of data held out for validation")
    p.add_argument("--seed",          type=int,   default=42)

    # Model
    p.add_argument("--base_model",    default=CLASSIFIER_BASE_MODEL)
    p.add_argument("--output_dir",    default=CLASSIFIER_OUTPUT_DIR)
    p.add_argument("--dtype",         default=CLASSIFIER_DTYPE,
                   choices=["bfloat16", "float32"])
    p.add_argument("--device",        default="cuda",
                   help="Torch device string (cuda / cpu / cuda:0)")

    # LoRA
    p.add_argument("--lora_r",        type=int,   default=LORA_R)
    p.add_argument("--lora_alpha",    type=int,   default=LORA_ALPHA)
    p.add_argument("--lora_dropout",  type=float, default=LORA_DROPOUT)
    p.add_argument("--lora_targets",  default=",".join(LORA_TARGET_MODULES),
                   help="Comma-separated LoRA target module names")

    # Training
    p.add_argument("--epochs",        type=int,   default=CLASSIFIER_EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=CLASSIFIER_BATCH_SIZE)
    p.add_argument("--grad_accum",    type=int,   default=CLASSIFIER_GRAD_ACCUM)
    p.add_argument("--lr",            type=float, default=CLASSIFIER_LR)
    p.add_argument("--warmup_ratio",  type=float, default=CLASSIFIER_WARMUP_RATIO)
    p.add_argument("--max_length",    type=int,   default=CLASSIFIER_MAX_LENGTH)
    p.add_argument("--threshold",     type=float, default=CLASSIFIER_THRESHOLD)

    # Checkpointing / resume
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint in --output_dir/checkpoint_last/ if it exists.",
    )

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    LOGGER.info("=== Qwen2.5-7B-Instruct + LoRA guardrail training ===")
    LOGGER.info("  base_model   : %s", args.base_model)
    LOGGER.info("  output_dir   : %s", args.output_dir)
    LOGGER.info("  dtype        : %s", args.dtype)
    LOGGER.info("  device       : %s", args.device)
    LOGGER.info("  LoRA r/α/d   : %d / %d / %.2f", args.lora_r, args.lora_alpha, args.lora_dropout)
    LOGGER.info("  LoRA targets : %s", args.lora_targets)
    LOGGER.info("  epochs       : %d", args.epochs)
    LOGGER.info("  batch_size   : %d  (grad_accum=%d → eff. batch=%d)",
                args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    LOGGER.info("  lr           : %.0e  warmup_ratio=%.2f", args.lr, args.warmup_ratio)
    LOGGER.info("  max_length   : %d  (truncation_side=left)", args.max_length)
    LOGGER.info("  threshold    : %.2f", args.threshold)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
