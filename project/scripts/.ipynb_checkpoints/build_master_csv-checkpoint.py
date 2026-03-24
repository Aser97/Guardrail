"""
scripts/build_master_csv.py
Assemble all generated/augmented CSV files into a single master dataset.

Steps:
1. Load seed_validation_set.csv texts so they can be excluded (these are
   evaluation-only conversations and must never appear in training data).
2. Load all source CSVs (generated_scratch, generated_camel, generated_camel_hard,
   esconv_preprocessed, augmented, generated_pair, generated_pair_hard,
   generated_gapfill — whichever exist).
3. Deduplicate by conversation text; also strip any row whose text matches
   a seed validation text.
4. Compute high_risk_any label from signals column (or use existing label column).
5. Apply class-balance strategy (optional upsampling).
6. Split into train.csv for the HuggingFace training script.
7. Print label distribution statistics.

Output:
    datasets/master.csv  — full deduplicated dataset (all columns)
    datasets/train.csv   — text + label + signals columns (for train_qwen_guardrail.py)
    datasets/submission_dataset.csv — hackathon deliverable schema

Usage:
    python project/scripts/build_master_csv.py [--balance] [--max_total 4000]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path


from config import (
    SIGNALS,
    SCRATCH_CSV, CAMEL_CSV, CAMEL_HARD_CSV,
    ESCONV_CSV, AUGMENTED_CSV,
    PAIR_CSV, PAIR_HARD_CSV,
    MASTER_CSV, TRAIN_CSV, SUBMISSION_CSV,
    DATASETS_DIR, SEED_VALIDATION_PATH,
)

# Gap-fill output (Phase 3 — generated only if gap analysis finds deficit signals)
_GAPFILL_CSV    = DATASETS_DIR / "generated_gapfill.csv"

# Note: degraded.csv (Phase 2e) and evolved.csv (Phase 2f) are intentionally
# excluded — see data_generation_strategy.md for rationale.

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

# CSV files to include (in order of generation phase)
# Phase 1a : from-scratch generation (high-risk + low-risk)
# Phase 1b : CAMEL dual-agent high-risk (standard + hard escalating/subtle tracks)
# ESConv   : preprocessed multi-turn emotional support (seeding source for 2a-2d)
# Phase 2a-2d: seed augmentation (lang_rewrite, persona_swap, signal_inject, signal_soften)
# Phase 2g : PAIR adversarial loop (positive + adversarial_negative)
# Phase 2g-hard: PAIR hard track (pair_hard_positive + pair_hard_subtle)
# Phase 3  : gap-fill targeted from-scratch (loaded only if it exists)
SOURCE_CSVS: list[Path] = [
    SCRATCH_CSV,       # Phase 1a
    CAMEL_CSV,         # Phase 1b  (standard)
    CAMEL_HARD_CSV,    # Phase 1b-hard (escalating + subtle tracks)
    ESCONV_CSV,        # ESConv preprocessed
    AUGMENTED_CSV,     # Phase 2a-2d (augmented seed rows)
    PAIR_CSV,          # Phase 2g   (PAIR adversarial loop)
    PAIR_HARD_CSV,     # Phase 2g-hard
    _GAPFILL_CSV,      # Phase 3    (gap-fill, if generated)
]

MASTER_FIELDNAMES = [
    "text", "label", "source", "primary_signal",
    "escalation_stage", "register", "language", "persona_id", "signals",
    "category",   # taxonomy category the conversation was generated for
]

# Fields required by the hackathon deliverable schema (seed_validation_set.csv)
SUBMISSION_FIELDNAMES = [
    "Conversation_id", "Turns", "Text", "Category", "Risk", "language", "label",
]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        LOGGER.info("Skipping %s (not found).", path)
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    LOGGER.info("Loaded %d rows from %s.", len(rows), path)
    return rows


def compute_label(row: dict) -> int:
    """
    Derive high_risk_any label.
    Priority: use 'label' column if present and valid, else derive from 'signals' JSON.
    """
    # Try explicit label column first
    label_str = row.get("label", "").strip()
    if label_str in ("0", "1"):
        return int(label_str)

    # Derive from signals JSON
    signals_str = row.get("signals", "")
    if signals_str:
        try:
            sig_dict = json.loads(signals_str)
            if any(int(sig_dict.get(s, 0)) == 1 for s in SIGNALS):
                return 1
            return 0
        except (json.JSONDecodeError, ValueError):
            pass

    # Default: unknown — mark as 0
    return 0


def load_seed_texts(path: Path) -> set[str]:
    """
    Load the seed validation set and return a set of normalised text strings.
    These are evaluation-only conversations and must be excluded from training data.
    Returns an empty set if the file does not exist (non-fatal).
    """
    if not path.exists():
        LOGGER.warning(
            "Seed validation set not found at %s — cannot guarantee seed exclusion.", path
        )
        return set()
    texts: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Seed CSV uses 'Text' (capital T); handle both casings
            t = row.get("Text", row.get("text", "")).strip()
            if t:
                texts.add(t)
    LOGGER.info("Loaded %d seed validation texts for exclusion from %s.", len(texts), path)
    return texts


def deduplicate(rows: list[dict], seed_texts: set[str] | None = None) -> list[dict]:
    """
    Remove duplicate texts (keep first occurrence).
    Also removes any row whose text exactly matches a seed validation conversation.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    n_seed_removed = 0
    for row in rows:
        text = row.get("text", "").strip()
        if not text:
            continue
        if seed_texts and text in seed_texts:
            n_seed_removed += 1
            continue
        if text not in seen:
            seen.add(text)
            unique.append(row)
    n_dup = len(rows) - len(unique) - n_seed_removed
    if n_seed_removed:
        LOGGER.warning(
            "Removed %d row(s) that matched seed validation set texts — "
            "these must not appear in training data.",
            n_seed_removed,
        )
    if n_dup > 0:
        LOGGER.info("Removed %d duplicate rows.", n_dup)
    return unique


def balance_classes(
    rows: list[dict],
    rng: random.Random,
    max_total: int = 0,
) -> list[dict]:
    """
    Upsample minority class so that label distribution is roughly 50/50.
    If max_total > 0, also cap the final size.
    """
    high  = [r for r in rows if compute_label(r) == 1]
    low   = [r for r in rows if compute_label(r) == 0]

    LOGGER.info(
        "Pre-balance: high_risk=%d low_risk=%d total=%d",
        len(high), len(low), len(rows),
    )

    # Upsample smaller class
    target_n = max(len(high), len(low))
    if len(high) < len(low):
        high = high + rng.choices(high, k=target_n - len(high))
    elif len(low) < len(high):
        low = low + rng.choices(low, k=target_n - len(low))

    balanced = high + low
    rng.shuffle(balanced)

    if max_total > 0 and len(balanced) > max_total:
        balanced = balanced[:max_total]

    LOGGER.info(
        "Post-balance: high_risk=%d low_risk=%d total=%d",
        sum(1 for r in balanced if compute_label(r) == 1),
        sum(1 for r in balanced if compute_label(r) == 0),
        len(balanced),
    )
    return balanced


def print_stats(rows: list[dict]) -> None:
    labels = [compute_label(r) for r in rows]
    high  = sum(1 for l in labels if l == 1)
    low   = sum(1 for l in labels if l == 0)
    total = len(labels)

    sources: dict[str, int] = {}
    for r in rows:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    langs: dict[str, int] = {}
    for r in rows:
        lang = r.get("language", "unknown")
        langs[lang] = langs.get(lang, 0) + 1

    print(f"\n{'─'*50}")
    print(f"MASTER DATASET STATISTICS")
    print(f"{'─'*50}")
    print(f"Total rows : {total}")
    print(f"High-risk  : {high}  ({100*high/max(1,total):.1f}%)")
    print(f"Low-risk   : {low}   ({100*low/max(1,total):.1f}%)")
    print(f"\nBy source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:<30} {count}")
    print(f"\nBy language:")
    for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
        print(f"  {lang:<10} {count}")
    print(f"{'─'*50}\n")


def save_master(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in MASTER_FIELDNAMES}
            out["label"] = compute_label(row)
            writer.writerow(out)
    LOGGER.info("Saved master CSV: %s (%d rows)", path, len(rows))


_SIGNALS_PATH_JSON       = "json_column"        # clean: signals JSON column used
_SIGNALS_PATH_INDIVIDUAL = "individual_columns"  # clean: per-signal columns used
_SIGNALS_PATH_FALLBACK   = "binary_fallback"     # problematic for high-risk rows


def _extract_signals_json(row: dict) -> tuple[str, str]:
    """
    Build a signals JSON string (dict keyed by SIGNALS) for a training row.
    Returns (json_string, path_taken) where path_taken is one of the
    _SIGNALS_PATH_* constants above.

    Priority order:
    1. Existing 'signals' column — JSON dict or aligned list; normalised to a dict.
    2. Individual signal columns (e.g. row['hopelessness'] = '1') — augmented.csv layout.
    3. Binary fallback: label=1 → every signal=1, label=0 → every signal=0.
       This is only correct for low-risk rows; for high-risk rows it is wrong.
    """
    # --- Priority 1: existing signals JSON column ---
    sig_str = row.get("signals", "").strip()
    if sig_str and sig_str not in ("{}", "[]", "null", ""):
        try:
            obj = json.loads(sig_str)
            if isinstance(obj, dict):
                return json.dumps({s: int(obj.get(s, 0)) for s in SIGNALS}), _SIGNALS_PATH_JSON
            if isinstance(obj, list) and len(obj) == len(SIGNALS):
                return json.dumps({s: int(v) for s, v in zip(SIGNALS, obj)}), _SIGNALS_PATH_JSON
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- Priority 2: individual signal columns (augmented.csv layout) ---
    sig_dict: dict[str, int] = {}
    found_any = False
    for s in SIGNALS:
        val = row.get(s, "").strip()
        if val in ("0", "1"):
            sig_dict[s] = int(val)
            found_any = True
        else:
            sig_dict[s] = 0
    if found_any:
        return json.dumps(sig_dict), _SIGNALS_PATH_INDIVIDUAL

    # --- Priority 3: binary fallback ---
    binary = compute_label(row)
    return json.dumps({s: binary for s in SIGNALS}), _SIGNALS_PATH_FALLBACK


def save_train(rows: list[dict], path: Path) -> None:
    """
    Write train.csv with three columns: text, label, signals.

    The 'signals' column is a JSON dict keyed by every signal name in SIGNALS,
    with 0/1 integer values.  train_qwen_guardrail.py reads this column to build
    per-head multi-label targets; without it the training script falls back to
    replicating the binary label across all 9 heads, discarding all annotation work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Counters split by path and label so we report only what matters
    counts: dict[str, int] = {
        _SIGNALS_PATH_JSON:       0,
        _SIGNALS_PATH_INDIVIDUAL: 0,
        _SIGNALS_PATH_FALLBACK:   0,
    }
    # Among fallback rows, track how many are high-risk (the only concerning ones)
    fallback_high_risk: list[str] = []   # store text previews for the log

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "signals"])
        writer.writeheader()
        for row in rows:
            lbl               = compute_label(row)
            sig_json, path_id = _extract_signals_json(row)
            counts[path_id]  += 1
            if path_id == _SIGNALS_PATH_FALLBACK and lbl == 1:
                preview = row.get("text", "")[:80].replace("\n", " ")
                fallback_high_risk.append(preview)
            writer.writerow({
                "text":    row.get("text", ""),
                "label":   lbl,
                "signals": sig_json,
            })

    total = len(rows)
    LOGGER.info(
        "Saved train.csv: %s  (%d rows total)", path, total,
    )
    LOGGER.info(
        "  Signal source breakdown:"
        "\n    %-22s %d  (signals JSON column present — correct)"
        "\n    %-22s %d  (per-signal columns used — correct)"
        "\n    %-22s %d  (binary fallback, low-risk only — expected and harmless)",
        "json_column",       counts[_SIGNALS_PATH_JSON],
        "individual_columns", counts[_SIGNALS_PATH_INDIVIDUAL],
        "binary_fallback (label=0)", counts[_SIGNALS_PATH_FALLBACK] - len(fallback_high_risk),
    )

    if fallback_high_risk:
        LOGGER.warning(
            "  *** CONCERNING: %d high-risk row(s) used binary fallback — "
            "no signal annotations found. All 9 heads will be trained as ON for these rows. "
            "Check that fix_signal_annotation.py was run on the source CSV.",
            len(fallback_high_risk),
        )
        for i, preview in enumerate(fallback_high_risk, 1):
            LOGGER.warning("    [%d] %.80s…", i, preview)
    else:
        LOGGER.info(
            "  *** OK: 0 high-risk rows used binary fallback — "
            "all positive labels have proper per-signal annotations.",
        )


def _count_turns(text: str) -> int:
    """Count conversation turns by counting 'User:' / 'user:' prefixes."""
    return max(1, text.lower().count("user:"))


def save_submission_csv(rows: list[dict], path: Path) -> None:
    """
    Write the deliverable dataset in the hackathon-required schema:
        Conversation_id | Turns | Text | Category | Risk | language | label

    Maps from internal master.csv columns:
        text           → Text
        category       → Category  (taxonomy category tracked at generation time)
        language       → language
        label (0/1)    → label AND Risk ("high"/"low")
        sequential int → Conversation_id
        turn count     → Turns (derived from text)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDNAMES)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            lbl = compute_label(row)
            text = row.get("text", "")
            writer.writerow({
                "Conversation_id": i,
                "Turns":           _count_turns(text),
                "Text":            text,
                "Category":        row.get("category", ""),
                "Risk":            "high" if lbl == 1 else "low",
                "language":        row.get("language", ""),
                "label":           lbl,
            })
    LOGGER.info("Saved submission CSV: %s (%d rows)", path, len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble master training CSV")
    parser.add_argument(
        "--balance",    action="store_true",
        help="Upsample minority class to 50/50",
    )
    parser.add_argument(
        "--max_total", type=int, default=0,
        help="Cap total dataset size after balancing (0 = no cap)",
    )
    parser.add_argument(
        "--seed",       type=int, default=42,
        help="Random seed for balancing/shuffling",
    )
    parser.add_argument(
        "--no_train_csv", action="store_true",
        help="Skip writing train.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load seed validation texts — these must be kept out of training data
    seed_texts = load_seed_texts(SEED_VALIDATION_PATH)

    # Load all sources
    all_rows: list[dict] = []
    for path in SOURCE_CSVS:
        all_rows.extend(load_csv(path))

    if not all_rows:
        LOGGER.error(
            "No source CSVs found. Run generate_scratch.py, generate_camel.py, "
            "and/or preprocess_esconv.py first."
        )
        return 1

    # Deduplicate + remove any seed validation rows
    rows = deduplicate(all_rows, seed_texts=seed_texts)

    # Balance (optional)
    if args.balance:
        rows = balance_classes(rows, rng, max_total=args.max_total)
    elif args.max_total > 0:
        rng.shuffle(rows)
        rows = rows[:args.max_total]

    print_stats(rows)

    save_master(rows, MASTER_CSV)

    if not args.no_train_csv:
        save_train(rows, TRAIN_CSV)

    save_submission_csv(rows, SUBMISSION_CSV)

    return 0


if __name__ == "__main__":
    sys.exit(main())
