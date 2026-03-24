"""
scripts/preprocess_esconv.py
Preprocess the ESConv dataset (Liu et al., ACL 2021) into a flat CSV suitable
for inclusion in our training pipeline.

ESConv source:
    https://github.com/thu-coai/Emotional-Support-Conversation
    Download: esconv_data.zip → esconv.json  (list of dialog objects)

Input format (esconv.json):
    [
      {
        "emotion_type": "anxiety",
        "problem_type": "job crisis",
        "situation": "...",
        "survey_score": {...},
        "dialog": [
          {"speaker": "usr", "content": "...", "annotation": {...}},
          {"speaker": "sys", "content": "...", "strategy": "..."},
          ...
        ]
      },
      ...
    ]

Output columns (esconv_preprocessed.csv):
    text            – full dialog flattened as  "User: ... \nSupport: ... \nUser: ..."
    label           – 0 or 1  (high_risk_any: 1 for problem types in ESCONV_HIGH_RISK_TYPES)
    source          – "esconv"
    problem_type    – original ESConv problem_type string
    emotion_type    – original ESConv emotion_type string
    n_turns         – number of dialog turns
    signals         – JSON array of triggered signals (empty for ESConv — no signal annotation)

Usage:
    python project/scripts/preprocess_esconv.py \\
        --input  datasets/esconv.json \\
        --output datasets/esconv_preprocessed.csv \\
        [--max_rows 400]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import csv
import logging
import sys
from pathlib import Path

# Allow running as  python project/scripts/preprocess_esconv.py

from config import (
    DATASETS_DIR,
    ESCONV_HIGH_RISK_TYPES,
    ESCONV_DEFAULT_LABEL,
)

# Maps ESConv problem_type keywords → our taxonomy category.
# Matching is substring/lowercase so partial matches work.
_ESCONV_CATEGORY_MAP: list[tuple[str, str]] = [
    ("suicide",          "Suicide"),
    ("self-harm",        "Self-Harm"),
    ("self harm",        "Self-Harm"),
    ("grief",            "Grief/Loss"),
    ("bereavement",      "Grief/Loss"),
    ("loss",             "Grief/Loss"),
    ("domestic",         "Safety & Abuse"),
    ("abuse",            "Safety & Abuse"),
    ("violence",         "Physical Violence"),
    ("bullying",         "Bullying/Harassment"),
    ("harassment",       "Bullying/Harassment"),
    ("substance",        "Substance Use"),
    ("alcohol",          "Substance Use"),
    ("drug",             "Substance Use"),
    ("school",           "School & Studies"),
    ("academic",         "School & Studies"),
    ("job",              "Seeking Support"),
    ("work",             "Seeking Support"),
    ("relationship",     "Romantic Relationships"),
    ("breakup",          "Romantic Relationships"),
    ("family",           "Home Life & Family"),
    ("isolation",        "Isolation"),
    ("lonely",           "Isolation"),
    ("loneliness",       "Isolation"),
    ("body",             "Body Image"),
    ("eating",           "Body Image"),
    ("identity",         "Identity & Belonging"),
    ("depression",       "Mental Health & Emotions"),
    ("anxiety",          "Mental Health & Emotions"),
    ("mental",           "Mental Health & Emotions"),
    ("emotion",          "Mental Health & Emotions"),
]


def _map_esconv_category(problem_type: str) -> str:
    """Return best-matching taxonomy category for an ESConv problem_type string."""
    pt = problem_type.strip().lower()
    for keyword, category in _ESCONV_CATEGORY_MAP:
        if keyword in pt:
            return category
    return "Seeking Support"   # sensible default for unmatched types

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def flatten_dialog(dialog: list[dict]) -> str:
    """
    Convert a list of ESConv dialog dicts into a single text string.

    Speaker mapping:
        "usr"  → "User"
        "sys"  → "Support"
        other  → capitalised as-is
    """
    SPEAKER_MAP = {"usr": "User", "sys": "Support"}
    lines: list[str] = []
    for turn in dialog:
        speaker = SPEAKER_MAP.get(turn.get("speaker", ""), turn.get("speaker", "Unknown"))
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def assign_label(problem_type: str) -> int:
    """Return 1 (high_risk) if problem_type overlaps with our signal space, else 0."""
    pt_lower = problem_type.strip().lower()
    for high_risk_type in ESCONV_HIGH_RISK_TYPES:
        if high_risk_type.lower() in pt_lower or pt_lower in high_risk_type.lower():
            return 1
    return ESCONV_DEFAULT_LABEL


def process_esconv(input_path: Path, max_rows: int = 0) -> list[dict]:
    """
    Load esconv.json and convert to list of row dicts.

    Parameters
    ----------
    input_path : Path
        Path to esconv.json (downloaded from ESConv repo).
    max_rows : int
        If > 0, cap the number of output rows.
    """
    LOGGER.info("Loading ESConv data from %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    LOGGER.info("Loaded %d ESConv conversations", len(data))

    rows: list[dict] = []
    skipped = 0

    for item in data:
        dialog   = item.get("dialog", [])
        if not dialog:
            skipped += 1
            continue

        text         = flatten_dialog(dialog)
        if not text.strip():
            skipped += 1
            continue

        problem_type = item.get("problem_type", "").strip()
        emotion_type = item.get("emotion_type", "").strip()
        label        = assign_label(problem_type)
        n_turns      = len(dialog)

        rows.append({
            "text":         text,
            "label":        label,
            "source":       "esconv",
            "problem_type": problem_type,
            "emotion_type": emotion_type,
            "n_turns":      n_turns,
            "signals":      "[]",   # ESConv has no per-signal annotation
            "category":     _map_esconv_category(problem_type),
        })

        if max_rows > 0 and len(rows) >= max_rows:
            LOGGER.info("Reached max_rows=%d, stopping early.", max_rows)
            break

    high_risk = sum(1 for r in rows if r["label"] == 1)
    LOGGER.info(
        "Processed %d rows (%d high_risk, %d low_risk). Skipped %d empty dialogs.",
        len(rows), high_risk, len(rows) - high_risk, skipped,
    )
    return rows


def save_csv(rows: list[dict], output_path: Path) -> None:
    """Write rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "label", "source", "problem_type", "emotion_type", "n_turns", "signals", "category"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d rows to %s", len(rows), output_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess ESConv dataset")
    parser.add_argument(
        "--input",
        default=str(DATASETS_DIR / "esconv.json"),
        help="Path to esconv.json (default: datasets/esconv.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DATASETS_DIR / "esconv_preprocessed.csv"),
        help="Output CSV path (default: datasets/esconv_preprocessed.csv)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=400,
        help="Maximum number of rows to output (0 = no limit, default 400)",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        LOGGER.error(
            "Input file not found: %s\n"
            "Download ESConv from https://github.com/thu-coai/Emotional-Support-Conversation",
            input_path,
        )
        return 1

    rows = process_esconv(input_path, max_rows=args.max_rows)
    if not rows:
        LOGGER.error("No rows produced — check input file format.")
        return 1

    save_csv(rows, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
