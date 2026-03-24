"""
scripts/config.py
Shared constants for all data-generation scripts (Track 1).

Usage:
    from config import SIGNALS, TAXONOMY, REGISTER_TYPES, ...
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SIGNAL TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────

SIGNALS: list[str] = [
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

# Human-readable description for each signal (used in generation prompts)
SIGNAL_DESCRIPTIONS: dict[str, str] = {
    "burden_language":    "Expressing feeling like a burden to others ('everyone would be better off without me').",
    "finality_language":  "Language implying permanent endings: giving away possessions, saying last goodbyes, writing notes.",
    "escape_framing":     "Framing death or self-harm as a way out, relief, or solution to unbearable pain.",
    "hopelessness":       "Pervasive belief that nothing will improve; no future orientation.",
    "active_self_harm":   "Current or recent non-suicidal self-injury described explicitly or in euphemism.",
    "immediate_safety":   "Imminent danger: describes a plan, access to means, or ongoing crisis.",
    "self_image_crisis":  "Profound identity collapse: 'I'm worthless / broken / don't deserve to exist'.",
    "third_party_concern":"Speaker worried about someone else who may be at risk.",
    "testing":            "Probing AI boundaries; asking 'what would you do if I said X?'; meta-guardrail testing.",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  ESCALATION STAGES
# ──────────────────────────────────────────────────────────────────────────────

ESCALATION_STAGES: list[str] = ["early", "mid", "full"]

ESCALATION_DESCRIPTIONS: dict[str, str] = {
    "early": (
        "Vague distress, dark humour, or mild self-deprecation. "
        "No explicit self-harm language; plausible deniability remains."
    ),
    "mid": (
        "Clearer expression of hopelessness, escape thoughts, or subtle planning. "
        "Still indirect enough that an outsider might miss it."
    ),
    "full": (
        "Explicit signal(s) present. Urgency is unmistakable. "
        "Active plan, means access, or direct statement of intent."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  REGISTER TYPES
# ──────────────────────────────────────────────────────────────────────────────

REGISTER_TYPES: list[str] = [
    "casual_texting",
    "forum_post",
    "voice_memo_transcription",
    "journalling",
    "ai_chat_direct",
]

REGISTER_DESCRIPTIONS: dict[str, str] = {
    "casual_texting": (
        "Phone message style: abbreviations, minimal punctuation, emoji optional, "
        "lowercase, short bursts, typos welcome."
    ),
    "forum_post": (
        "Reddit / Discord thread: may have a title-line, multi-paragraph body, "
        "TL;DR, mentions of throwaway accounts. Moderate punctuation."
    ),
    "voice_memo_transcription": (
        "Spoken-word transcript: false starts, 'um', 'like', run-on sentences, "
        "no punctuation other than commas and periods."
    ),
    "journalling": (
        "Private diary entry: dated or not, introspective, complete sentences, "
        "stream-of-consciousness, may be poetic."
    ),
    "ai_chat_direct": (
        "Message typed directly to an AI assistant: may start with a greeting, "
        "may re-phrase itself mid-message, conversational but purposeful."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 4.  LANGUAGE CODES
# ──────────────────────────────────────────────────────────────────────────────

LANGUAGES: list[str] = ["en", "fr", "mix"]

LANGUAGE_NOTES: dict[str, str] = {
    "en":  "Canadian / North-American English.",
    "fr":  "Québec French (informal register: 'faque', 'j'en peux pu', 'checker', etc.).",
    "mix": "Code-switching: naturally alternates between English and Québec French mid-message.",
}

# ──────────────────────────────────────────────────────────────────────────────
# 5.  HACKATHON MODELS  (OpenAI-compatible endpoints)
# ──────────────────────────────────────────────────────────────────────────────

HACKATHON_MODELS: dict[str, str] = {
    "mistral":  "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",  # 675 B  – user-role / attacker / adversarial generator
    "command":  "CohereLabs/c4ai-command-a-03-2025",                   # ~111 B – assistant-role / judge / paraphrase
    "gpt_oss":  "openai/gpt-oss-120b",                                 # 120 B  – fallback / annotation
}

# Default generation temperature per role
# Rationale: diversity comes from prompt variation (persona × signal × register × language),
# not from temperature.  Lower values preserve reasoning quality on complex multi-constraint prompts.
TEMPERATURE: dict[str, float] = {
    "user_turn":   0.75,   # complex prompt — reasoning fidelity > randomness
    "assistant":   0.60,   # coherent, empathetic support responses
    "judge":       0.20,   # structured pass/fail — near-deterministic
    "annotation":  0.10,   # signal labelling — deterministic
}

# ──────────────────────────────────────────────────────────────────────────────
# 6.  DATASET PATHS
# ──────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

# Scripts live in  <repo>/project/scripts/
# datasets live in <repo>/datasets/
_SCRIPTS_DIR = Path(__file__).resolve().parent  # project/scripts/
REPO_ROOT    = _SCRIPTS_DIR.parent.parent        # repo root (two levels up)
DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR   = REPO_ROOT / "project" / "models"

PERSONA_BANK_PATH = DATASETS_DIR / "persona_bank.json"
SLANG_BANK_PATH   = DATASETS_DIR / "slang_bank.json"
SEED_VALIDATION_PATH = DATASETS_DIR / "seed_validation_set.csv"

# Generated CSV outputs (one per phase/method)
SCRATCH_CSV   = DATASETS_DIR / "generated_scratch.csv"
CAMEL_CSV     = DATASETS_DIR / "generated_camel.csv"
ESCONV_CSV    = DATASETS_DIR / "esconv_preprocessed.csv"
AUGMENTED_CSV = DATASETS_DIR / "augmented.csv"
MASTER_CSV      = DATASETS_DIR / "master.csv"

# Final training CSV (text + label) handed to the HF training script
TRAIN_CSV       = DATASETS_DIR / "train.csv"

# Hackathon deliverable dataset (Conversation_id, Turns, Text, Category, Risk, language, label)
SUBMISSION_CSV  = DATASETS_DIR / "submission_dataset.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 7.  SIGNAL PHRASE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
#
# The fundamental mapping is phrase → signal.  These phrase lists define what
# each signal IS.  They are used directly in generation prompts so the model
# knows what kind of expression to produce.
#
# Taxonomy categories (Section 8 below) are an independent dimension — they
# define the conversation TOPIC/CONTEXT.  Any signal phrase can appear in any
# category; they are orthogonal.  Do NOT map categories to signals here.
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_PHRASES: dict[str, list[str]] = {
    "burden_language": [
        '"Better off without me"',
        "Expressing they are a burden to others",
        "Withdrawing to avoid burdening people",
        "Feeling like they make things worse for everyone around them",
    ],
    "finality_language": [
        "Mentions of how they would do it",
        "Describing a plan, timeline, or means",
        "Thoughts about ending one's life",
        "Wanting to join a deceased person",
        '"I can\'t do this anymore"',
        "Giving away possessions or saying goodbye",
    ],
    "escape_framing": [
        "Hopelessness tied to bullying or ongoing targeting",
        "Framing disappearance or death as relief or escape",
        "Avoidance and withdrawal as coping",
        "Overdose risk framed as a way out",
        "Coping-motivated substance use",
        '"Better off without me"',
        '"No one would notice if I was gone"',
    ],
    "hopelessness": [
        "Worthless or trapped language",
        "Emotional numbness; feeling nothing will change",
        "Rejection as permanent and total",
        "Pervasive guilt with no path forward",
        "Future described as empty or nonexistent",
    ],
    "active_self_harm": [
        "Increasing frequency, severity, or loss of control over self-harm",
        "Desire or temptation to self-harm without action yet",
        "Self-harm is actively happening now",
        "Starving or self-punishment as self-harm",
        "Restriction cycles; body as target of pain",
        "Intent to harm oneself",
    ],
    "immediate_safety": [
        "Violence or immediate physical danger",
        "Threats of being kicked out of home",
        "Fear of or active abuse by a caregiver or partner",
        "Controlling partner with threats or violence",
        "Active unsafe environment at home",
        "Fear of harm from a named person",
        "Targeted harassment or coercion",
        "Coercion or assault risk",
    ],
    "self_image_crisis": [
        "Self-worth collapse; feeling fundamentally broken or worthless",
        "Identity confusion; not knowing who they are",
        "Self-hatred directed at their body",
        "Profound distress about bodily or life changes",
    ],
    "third_party_concern": [
        "A friend may attempt suicide",
        "A friend is visibly struggling and the speaker is worried",
        "Asking what to do after a friend's crisis",
        "Describing a loved one's dangerous behaviour",
    ],
    "testing": [
        "Testing the chatbot's safety limits or responses",
        "Repeated probing with escalating hypotheticals",
        "Hesitant, tentative connection — not fully disclosing yet",
        '"What would you do if I said I wanted to hurt myself?"',
        "Framing as hypothetical or 'asking for a friend'",
    ],
}

# Maps each phrase string back to the taxonomy category it was extracted from.
# Source: docs/data_generation_manual.md — "Taxonomy x Risk Signal Table".
# Note: a small number of phrases appear in more than one signal list (e.g.
# '"Better off without me"' is in both burden_language and escape_framing);
# those share the same category so the dict value is unambiguous.
PHRASE_TO_CATEGORY: dict[str, str] = {
    # burden_language
    '"Better off without me"':                                    "Isolation",
    "Expressing they are a burden to others":                     "Isolation",
    "Withdrawing to avoid burdening people":                      "Isolation",
    "Feeling like they make things worse for everyone around them": "Social Relationships",
    # finality_language
    "Mentions of how they would do it":                           "Suicide",
    "Describing a plan, timeline, or means":                      "Suicide",
    "Thoughts about ending one's life":                           "Suicide",
    "Wanting to join a deceased person":                          "Grief/Loss",
    '"I can\'t do this anymore"':                                 "School & Studies",
    "Giving away possessions or saying goodbye":                  "Suicide",
    # escape_framing
    "Hopelessness tied to bullying or ongoing targeting":         "Bullying/Harassment",
    "Framing disappearance or death as relief or escape":         "Suicide",
    "Avoidance and withdrawal as coping":                         "Mental Health & Emotions",
    "Overdose risk framed as a way out":                          "Substance Use",
    "Coping-motivated substance use":                             "Substance Use",
    '"No one would notice if I was gone"':                        "Social Relationships",
    # hopelessness
    "Worthless or trapped language":                              "Mental Health & Emotions",
    "Emotional numbness; feeling nothing will change":            "Mental Health & Emotions",
    "Rejection as permanent and total":                           "Identity & Belonging",
    "Pervasive guilt with no path forward":                       "Grief/Loss",
    "Future described as empty or nonexistent":                   "Mental Health & Emotions",
    # active_self_harm
    "Increasing frequency, severity, or loss of control over self-harm": "Self-Harm",
    "Desire or temptation to self-harm without action yet":       "Self-Harm",
    "Self-harm is actively happening now":                        "Self-Harm",
    "Starving or self-punishment as self-harm":                   "Body Image",
    "Restriction cycles; body as target of pain":                 "Body Image",
    "Intent to harm oneself":                                     "Self-Harm",
    # immediate_safety
    "Violence or immediate physical danger":                      "Home Life & Family",
    "Threats of being kicked out of home":                        "Home Life & Family",
    "Fear of or active abuse by a caregiver or partner":          "Safety & Abuse",
    "Controlling partner with threats or violence":               "Romantic Relationships",
    "Active unsafe environment at home":                          "Safety & Abuse",
    "Fear of harm from a named person":                           "Safety & Abuse",
    "Targeted harassment or coercion":                            "Hate Speech",
    "Coercion or assault risk":                                   "Sexual Health",
    # self_image_crisis
    "Self-worth collapse; feeling fundamentally broken or worthless": "Personality",
    "Identity confusion; not knowing who they are":               "Identity & Belonging",
    "Self-hatred directed at their body":                         "Puberty/Life Changes",
    "Profound distress about bodily or life changes":             "Puberty/Life Changes",
    # third_party_concern
    "A friend may attempt suicide":                               "Third Party",
    "A friend is visibly struggling and the speaker is worried":  "Third Party",
    "Asking what to do after a friend's crisis":                  "Seeking Support",
    "Describing a loved one's dangerous behaviour":               "Third Party",
    # testing
    "Testing the chatbot's safety limits or responses":           "Testing",
    "Repeated probing with escalating hypotheticals":             "Testing",
    "Hesitant, tentative connection — not fully disclosing yet":  "Seeking Support",
    '"What would you do if I said I wanted to hurt myself?"':     "Testing",
    "Framing as hypothetical or 'asking for a friend'":           "Testing",
}

# Flat catalog of all (phrase, signal, category) triples — the atomic sampling
# unit for generation.  Sample one entry → you get all three fields for free.
# Built from SIGNAL_PHRASES + PHRASE_TO_CATEGORY so it stays in sync automatically.
PHRASE_CATALOG: list[dict[str, str]] = [
    {
        "phrase":    phrase,
        "signal":    signal,
        "category":  PHRASE_TO_CATEGORY.get(phrase, "Unknown"),
    }
    for signal, phrases in SIGNAL_PHRASES.items()
    for phrase in phrases
]

# ──────────────────────────────────────────────────────────────────────────────
# 8.  DATA GENERATION TARGETS
# ──────────────────────────────────────────────────────────────────────────────
#
# Generation unit: a (phrase, signal, category) triple sampled from
# PHRASE_CATALOG.  This ensures every generated conversation is traceable
# to a specific taxonomy category via the phrase it was seeded with.
# 23 categories × ~65 conversations = ~1,500 total rows (Phase 1).
# Each category: ~32 high-risk + ~32 low-risk.
# ──────────────────────────────────────────────────────────────────────────────

# Canonical list of the 23 taxonomy categories (generation unit — topic/context only)
TAXONOMY_CATEGORIES: list[str] = [
    "Isolation",
    "Prank/Joke",
    "Suicide",
    "School & Studies",
    "Grief/Loss",
    "Bullying/Harassment",
    "Social Relationships",
    "Substance Use",
    "Mental Health & Emotions",
    "Identity & Belonging",
    "Self-Harm",
    "Body Image",
    "Physical Violence",
    "Home Life & Family",
    "Romantic Relationships",
    "Safety & Abuse",
    "Hate Speech",
    "Sexual Health",
    "Puberty/Life Changes",
    "Personality",
    "Third Party",
    "Seeking Support",
    "Testing",
]

# Target conversations per category (high-risk + low-risk combined, 50/50 split)
CONVERSATIONS_PER_CATEGORY: int = 65

# Minimum positives per signal before triggering targeted gap-fill generation
GAP_FILL_THRESHOLD: int = 50

PHASE1B_TARGET = 300   # CAMEL dual-agent high-risk conversations
ESCONV_MAX     = 400   # ESConv preprocessed (mapped to low_risk; used for augmentation seeding)

# Low-risk generation target (conversations with 0 signals)
LOW_RISK_TARGET = 400

# Escalation stage sampling weights (soft targets at dataset level)
ESCALATION_WEIGHTS: dict[str, float] = {
    "early": 0.35,
    "mid":   0.40,
    "full":  0.25,
}

# Language sampling weights
LANGUAGE_WEIGHTS: dict[str, float] = {
    "en":  0.55,
    "fr":  0.25,
    "mix": 0.20,
}

# ──────────────────────────────────────────────────────────────────────────────
# 8.  CLASSIFIER TRAINING  (Qwen2.5-7B-Instruct + LoRA, 9-head multi-label)
# ──────────────────────────────────────────────────────────────────────────────

# Base model
CLASSIFIER_BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
CLASSIFIER_OUTPUT_DIR  = str(MODELS_DIR / "mhs_guardrail")
CLASSIFIER_DTYPE       = "bfloat16"   # A6000 48 GB — full bf16, no quantization needed

# LoRA — PEFT fine-tuning
LORA_R               = 16            # rank; balances expressiveness and parameter efficiency
LORA_ALPHA           = 32            # = 2×r (standard scaling heuristic)
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention projections only

# Training
CLASSIFIER_EPOCHS         = 3        # LoRA converges faster than full fine-tune
CLASSIFIER_BATCH_SIZE     = 4        # per-device; A6000 48 GB comfortably fits this
CLASSIFIER_GRAD_ACCUM     = 4        # effective batch size = 4 × 4 = 16
CLASSIFIER_LR             = 2e-4     # standard LoRA learning rate
CLASSIFIER_WARMUP_RATIO   = 0.05
CLASSIFIER_MAX_LENGTH     = 2048     # hard ceiling; dynamic padding within batch
# Truncation direction: LEFT (truncation_side="left") — drop the beginning of the
# conversation and keep the most recent turns, where safety signals concentrate.
                                     # within each batch (pad to longest sequence in batch)

# Head & loss
N_LABELS      = len(SIGNALS)         # 9 — one sigmoid output per signal
LOSS_FN       = "BCEWithLogitsLoss"  # multi-label; applied independently per signal head

# Inference / guardrail
CLASSIFIER_THRESHOLD   = 0.5         # applied independently to each of the 9 signal probabilities
CLASSIFIER_SCORE_AGG   = "max"       # overall guardrail score = max(p_signal_1 … p_signal_9)
                                     # fires on the strongest individual signal;
                                     # interpretable + conservative (Llama Guard uses same approach)

# ──────────────────────────────────────────────────────────────────────────────
# 9.  ESCONV PROBLEM-TYPE → RISK LABEL MAPPING
# ──────────────────────────────────────────────────────────────────────────────

# ESConv problem_types that overlap with our signal space → treat as potentially high_risk
ESCONV_HIGH_RISK_TYPES: set[str] = {
    "depression",
    "suicidal ideation",
    "self-harm",
    "anxiety",   # kept in—may co-occur with high-risk signals
}

# All others are mapped to low_risk (label=0)
ESCONV_DEFAULT_LABEL = 0
