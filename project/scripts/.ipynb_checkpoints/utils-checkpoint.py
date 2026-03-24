"""
scripts/utils.py
Shared utilities for the data generation pipeline.

persona_to_text()  — converts a persona dict into a prose paragraph
                     covering personal background only (who this person is).
                     Language, slang, and stressor remain separate prompt
                     sections and must NOT be folded in here.

stressor_to_text() — renders the stressor_context sub-dict as a single
                     prose sentence for its own dedicated prompt section.

init_csv()         — open (or reopen) a CSV output file for incremental
                     writing; returns (file_handle, DictWriter).

append_row()       — write a single row to an already-open DictWriter and
                     immediately flush so no data is lost on interruption.

count_csv_rows()   — count rows in an existing CSV (excluding header),
                     used by --append / resume logic.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import IO, Any


def init_csv(
    path: Path,
    fieldnames: list[str],
    append: bool = False,
) -> tuple[IO[str], "csv.DictWriter[str]"]:
    """
    Open *path* for incremental row writing.

    Parameters
    ----------
    path     : output file path (parent dirs created automatically)
    fieldnames : CSV column names
    append   : if True and file exists, open in append mode (no header written);
               if False or file does not exist, open in write mode (header written)

    Returns
    -------
    (file_handle, writer)
        Keep both alive for the duration of generation; call file_handle.close()
        when done.  Pass *writer* to append_row().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        fh: IO[str] = open(path, "a", newline="", encoding="utf-8")
        writer: csv.DictWriter[str] = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore"
        )
        # Header already present — do not write again
    else:
        fh = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
    return fh, writer


def append_row(writer: "csv.DictWriter[str]", fh: IO[str], row: dict[str, Any]) -> None:
    """
    Write *row* via *writer* and flush immediately.

    Flushing after every row ensures that interruptions (Paperspace 6 h
    session limit, OOM kills, keyboard interrupts) do not lose already-
    generated data.
    """
    writer.writerow(row)
    fh.flush()


def count_csv_rows(path: Path) -> int:
    """Return the number of data rows in *path* (0 if file does not exist)."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            next(reader)   # skip header
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def count_csv_rows_by_field(path: Path, field: str) -> dict[str, int]:
    """
    Return a counter {field_value: row_count} for *field* in *path*.

    Used by generate_scratch.py --append to compute per-signal existing counts.
    Returns {} if file does not exist.
    """
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(field, "")
            counts[val] = counts.get(val, 0) + 1
    return counts


def vprint(verbose: bool, *args: Any, sep: str = "\n", **kwargs: Any) -> None:
    """
    Print only when verbose=True.  Each positional arg is separated by *sep*
    (default newline) to make multi-section debug output easy to read.
    """
    if verbose:
        print(sep.join(str(a) for a in args), **kwargs)
        sys.stdout.flush()


def persona_to_text(persona: dict) -> str:
    """
    Convert a persona dict into a natural-language paragraph the LLM can
    write FROM — not a reformatted dictionary.

    Covers personal characterisation only: age, location, occupation, family
    status, mental health background, substance use, identity axes, and voice
    register notes.

    Deliberately excludes stressor_context (handled by stressor_to_text),
    language, and slang — those stay as separate, distinct prompt sections.

    Parameters
    ----------
    persona : dict
        A single entry from datasets/persona_bank.json.

    Returns
    -------
    str
        A 2–4 sentence prose description ready for prompt insertion.
    """
    age        = persona.get("age", "?")
    gender     = persona.get("gender", "person")
    region     = persona.get("region", "Canada")
    occupation = persona.get("occupation", "")
    family     = persona.get("family_status", "")
    mh_bg      = persona.get("mental_health_background")
    substance  = persona.get("substance_use")
    identities = persona.get("identity_axes") or []
    voice      = persona.get("voice_notes", "")

    # ── Sentence 1: who they are ─────────────────────────────────────────────
    pronoun, verb_be = _pronoun_verb(gender)
    occupation_clause = f", currently a {occupation}" if occupation and occupation != "varies" else ""
    family_clause = f", and {family}" if family else ""
    s1 = f"{pronoun.capitalize()} {verb_be} {age} years old and based in {region}{occupation_clause}{family_clause}."

    # ── Sentence 2: mental health and substance use (if present) ─────────────
    s2_parts: list[str] = []
    if mh_bg and mh_bg not in ("no known diagnosis", "none"):
        s2_parts.append(mh_bg)
    if substance and substance not in ("none", "None"):
        s2_parts.append(f"context of {substance}")
    s2 = ""
    if s2_parts:
        s2 = f"Their health background includes: {'; '.join(s2_parts)}."

    # ── Sentence 3: identity (if non-empty) ──────────────────────────────────
    s3 = ""
    if identities:
        id_str = ", ".join(identities)
        s3 = f"They identify as: {id_str}."

    # ── Sentence 4: voice / register notes ───────────────────────────────────
    s4 = voice if voice else ""

    paragraph = " ".join(s for s in [s1, s2, s3, s4] if s)
    return paragraph.strip()


def stressor_to_text(persona: dict) -> str:
    """
    Render the persona's stressor_context as a single prose sentence.

    Returns an empty string if no stressors are present, so callers can
    conditionally include the stressor section.

    Parameters
    ----------
    persona : dict
        A single entry from datasets/persona_bank.json.

    Returns
    -------
    str
        One sentence describing current stressors, or "" if none.
    """
    stressor = persona.get("stressor_context") or {}
    parts: list[str] = []
    if stressor.get("personal"):
        parts.append(stressor["personal"])
    if stressor.get("relational"):
        parts.append(stressor["relational"])
    if stressor.get("world"):
        parts.append(stressor["world"])
    if not parts:
        return ""
    return f"Right now they are dealing with: {'; '.join(parts)}."


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pronoun_verb(gender: str) -> tuple[str, str]:
    """Return (pronoun, conjugated 'to be') for the persona's gender."""
    g = gender.lower()
    if g in ("female", "woman", "girl", "trans woman"):
        return "she", "is"
    if g in ("male", "man", "boy"):
        return "he", "is"
    return "they", "are"   # non-binary, other, unknown
