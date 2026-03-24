"""
scripts/degrade_register.py
Phase 2e — Linguistic realism degradation (rule-based, no LLM, free).

Applies four stochastic transformations to user turns only:
  1. Fragmentation   — break complete sentences; add trailing "..." or abrupt stops
  2. Abbreviation    — replace spelled-out phrases with age-appropriate slang abbreviations
  3. Affect flatten  — strip emphasis markers; replace strong adjectives with minimizers
  4. Typo injection  — probabilistic adjacent-key swaps / drops / duplicates

All transformations are applied to user turns only.
Assistant turns and all label/signal columns are preserved unchanged.

Usage:
    python project/scripts/degrade_register.py \\
        --inputs  datasets/generated_scratch.csv datasets/generated_camel.csv \\
        --output  datasets/degraded.csv \\
        --rate    0.30 \\
        --seed    42
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import csv
import random
import re
import sys
from pathlib import Path


from build_master_csv import MASTER_FIELDNAMES


# ──────────────────────────────────────────────────────────────────────────────
# Abbreviation lookup tables
# ──────────────────────────────────────────────────────────────────────────────

_ABBREV_EN: dict[str, str] = {
    "i don't know": "idk",
    "i do not know": "idk",
    "to be honest": "tbh",
    "not going to lie": "ngl",
    "not gonna lie": "ngl",
    "for real": "fr",
    "in my opinion": "imo",
    "oh my god": "omg",
    "by the way": "btw",
    "as soon as possible": "asap",
    "i don't care": "idc",
    "what the hell": "wth",
    "what the fuck": "wtf",
    "laughing out loud": "lol",
    "oh my gosh": "omg",
    "rolling on the floor laughing": "lmao",
    "got to go": "gtg",
    "talk to you later": "ttyl",
    "never mind": "nvm",
    "right now": "rn",
    "because": "bc",
    "without": "w/o",
    "with": "w/",
    "you": "u",
    "are": "r",
    "okay": "ok",
    "please": "pls",
    "something": "sth",
    "everyone": "everyone",
    "probably": "prolly",
    "definitely": "def",
    "obviously": "obv",
    "seriously": "srsly",
    "kind of": "kinda",
    "sort of": "sorta",
    "want to": "wanna",
    "going to": "gonna",
    "have to": "hafta",
    "supposed to": "sposta",
    "out of": "outta",
    "a lot": "alot",
    "right": "rly",
    "really": "rly",
    "actually": "actually",
}

_ABBREV_FR: dict[str, str] = {
    "je ne sais pas": "jspp",
    "j'en sais pas": "jspp",
    "franchement": "frnc",
    "vraiment": "vmt",
    "tellement": "tmt",
    "maintenant": "mtn",
    "quelque chose": "qqch",
    "quelqu'un": "qqun",
    "s'il te plaît": "stp",
    "s'il vous plaît": "svp",
    "parce que": "pcq",
    "en tout cas": "etc...",
    "tu sais": "tsé",
    "je suis": "chu",
    "c'est": "c",
    "pas du tout": "pdt",
    "ben là": "bnla",
    "faque": "faque",
    "j'en peux pu": "jppu",
    "je t'aime": "jvm",
    "pourquoi": "pkoi",
    "comment": "cmnt",
    "quand même": "qm",
}

# ──────────────────────────────────────────────────────────────────────────────
# Affect words to flatten
# ──────────────────────────────────────────────────────────────────────────────

_AFFECT_FLATTEN_EN: list[tuple[str, str]] = [
    (r"\breally\b", "kinda"),
    (r"\bvery\b", "kinda"),
    (r"\bextremely\b", "pretty"),
    (r"\bsuper\b", "kinda"),
    (r"\bso\s+much\b", "a lot i guess"),
    (r"\bso\b", "kinda"),
    (r"\bcompletely\b", "like"),
    (r"\babsolutely\b", "honestly"),
    (r"\bterrible\b", "bad"),
    (r"\bhorrible\b", "bad"),
    (r"\bdevastated\b", "upset"),
    (r"\bhopeless\b", "just... idk"),
    (r"\bdesperate\b", "kinda stuck"),
    (r"\bI feel\b", "i feel like"),
    (r"\bI am feeling\b", "idk i feel kinda"),
    (r"\bI'm feeling\b", "idk i feel kinda"),
    (r"!!+", "."),
    (r"!(?=[^!])", "."),
]

_AFFECT_FLATTEN_FR: list[tuple[str, str]] = [
    (r"\bvraiement\b", "genre"),
    (r"\bvraiment\b", "genre"),
    (r"\btellement\b", "assez"),
    (r"\bextrêmement\b", "pas mal"),
    (r"\bterrible\b", "pas le fun"),
    (r"\bhorrible\b", "weird"),
    (r"\bdésespéré\b", "tanné"),
    (r"\bdésespérée\b", "tannée"),
    (r"!!+", "."),
    (r"!(?=[^!])", "."),
]

# Adjacent key map for QWERTY typos
_ADJACENT_KEYS: dict[str, str] = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wrsdf",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "uojk", "j": "huikm n",
    "k": "jiol,m", "l": "kop;.", "m": "njk,", "n": "bhjm", "o": "iplk",
    "p": "ol;[", "q": "wa", "r": "edft", "s": "waedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

# ──────────────────────────────────────────────────────────────────────────────
# Individual transformation functions
# ──────────────────────────────────────────────────────────────────────────────

def _fragment(text: str, rng: random.Random) -> str:
    """Break some complete sentences into incomplete thoughts."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for sent in sentences:
        if len(sent) > 40 and rng.random() < 0.4:
            # Find a mid-point conjunction/comma to break at
            mid = len(sent) // 2
            break_candidates = [
                m.start() for m in re.finditer(r"\b(and|but|so|because|bc|cause|when|like|idk)\b", sent)
                if m.start() > 10
            ]
            if break_candidates:
                bp = min(break_candidates, key=lambda x: abs(x - mid))
                trail = rng.choice(["...", "...", " idk", ""])
                sent = sent[:bp].rstrip() + trail
        out.append(sent)
    return " ".join(out)


def _abbreviate(text: str, lang: str, rng: random.Random) -> str:
    """Replace spelled-out phrases with age-appropriate abbreviations."""
    abbrev_map = _ABBREV_FR if lang == "fr" else _ABBREV_EN
    result = text
    for phrase, abbr in abbrev_map.items():
        if rng.random() < 0.5 and phrase.lower() in result.lower():
            result = re.sub(re.escape(phrase), abbr, result, flags=re.IGNORECASE)
    return result


def _flatten_affect(text: str, lang: str, rng: random.Random) -> str:
    """Replace strong emotional words with minimizers; strip exclamation marks."""
    rules = _AFFECT_FLATTEN_FR if lang == "fr" else _AFFECT_FLATTEN_EN
    result = text
    for pattern, replacement in rules:
        if rng.random() < 0.6:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _inject_typos(text: str, age_group: str, rng: random.Random) -> str:
    """Probabilistic adjacent-key typos; higher rate for 13-16yo."""
    # Error rate by age group
    rate_map = {"7-12": 0.03, "13-16": 0.06, "17-22": 0.03, "23-33": 0.01}
    rate = rate_map.get(age_group, 0.03)
    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        roll = rng.random()
        ch_lower = ch.lower()
        if roll < rate and ch.isalpha() and ch_lower in _ADJACENT_KEYS:
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap":
                replacement = rng.choice(_ADJACENT_KEYS[ch_lower])
                result.append(replacement if ch.islower() else replacement.upper())
            elif op == "drop":
                pass  # skip this character
            else:  # dup
                result.append(ch)
                result.append(ch)
        else:
            result.append(ch)
        i += 1
    return "".join(result)


# ──────────────────────────────────────────────────────────────────────────────
# Turn-level processor
# ──────────────────────────────────────────────────────────────────────────────

def _degrade_user_turn(turn_text: str, lang: str, age_group: str, rng: random.Random) -> str:
    """Apply a random subset of transformations to a single user turn."""
    ops = rng.sample(["fragment", "abbreviate", "flatten", "typo"], k=rng.randint(1, 3))
    result = turn_text
    if "fragment" in ops:
        result = _fragment(result, rng)
    if "abbreviate" in ops:
        result = _abbreviate(result, lang, rng)
    if "flatten" in ops:
        result = _flatten_affect(result, lang, rng)
    if "typo" in ops:
        result = _inject_typos(result, age_group, rng)
    return result.strip()


def _degrade_conversation(text: str, lang: str, persona: str, rng: random.Random) -> str:
    """
    Apply degradation to user turns only within a full conversation string.
    Conversation format: 'user: ...\nassistant: ...\nuser: ...'
    """
    # Infer age_group from persona string (fallback: 17-22)
    age_group = "17-22"
    if persona:
        m = re.search(r"\b(\d{1,2})\b", persona)
        if m:
            age = int(m.group(1))
            if age <= 12:
                age_group = "7-12"
            elif age <= 16:
                age_group = "13-16"
            elif age <= 22:
                age_group = "17-22"
            else:
                age_group = "23-33"

    lines = text.split("\n")
    result = []
    for line in lines:
        if line.startswith("user:"):
            turn_body = line[len("user:"):].strip()
            turn_body = _degrade_user_turn(turn_body, lang, age_group, rng)
            result.append(f"user: {turn_body}")
        else:
            result.append(line)
    return "\n".join(result)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 2e: linguistic realism degradation (rule-based, no LLM)."
    )
    p.add_argument("--inputs",  nargs="+", required=True,
                   help="Input CSV paths (generated_scratch.csv, generated_camel.csv, ...)")
    p.add_argument("--output",  default="datasets/degraded.csv")
    p.add_argument("--rate",    type=float, default=0.30,
                   help="Fraction of rows to degrade (applied to each input file)")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    all_rows: list[dict] = []

    for inp_path in args.inputs:
        path = Path(inp_path)
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue
        rows = load_csv(path)
        n_degrade = int(len(rows) * args.rate)
        indices = rng.sample(range(len(rows)), k=min(n_degrade, len(rows)))
        degraded_count = 0
        for i, row in enumerate(rows):
            if i in indices:
                original_text = row.get("text", "")
                lang = row.get("language", "en")
                persona = row.get("persona", "")
                new_row = dict(row)
                new_row["text"] = _degrade_conversation(original_text, lang, persona, rng)
                new_row["source"] = row.get("source", "unknown") + "_degraded"
                new_row["conversation_id"] = row.get("conversation_id", "") + "_deg"
                all_rows.append(new_row)
                degraded_count += 1
        print(f"  {path.name}: {degraded_count}/{len(rows)} rows degraded.")

    save_csv(all_rows, Path(args.output))
    print(f"✓ Saved {len(all_rows)} degraded rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
