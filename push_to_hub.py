"""
push_to_hub.py
==============
Push the KHP Mental Health Safety Guardrail dataset to HuggingFace Hub.

Run from the root of the Guardrail-For-Agents repo on Paperspace:
    pip install datasets huggingface_hub --upgrade
    python push_to_hub.py
"""

import json
import random
import os
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import login, create_repo

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID   = "AserLompo/khp-youth-mental-health-guardrail"   # ← update if needed
MASTER_CSV   = "datasets/master.csv"
SEED         = 42
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# TEST_RATIO is the remainder (0.10)

# ── Login ─────────────────────────────────────────────────────────────────────
# Set HF_TOKEN in your environment or paste it here (do NOT commit the token)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "Set the HF_TOKEN environment variable before running:\n"
        "  export HF_TOKEN=hf_..."
    )
login(token=hf_token)

# ── Create repo if it doesn't exist ───────────────────────────────────────────
create_repo(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    private=False,
    exist_ok=True,      # no error if it already exists
)
print(f"Repo ready: https://huggingface.co/datasets/{HF_REPO_ID}")

# ── Load & flatten ─────────────────────────────────────────────────────────────
df = pd.read_csv(MASTER_CSV, dtype_backend="numpy_nullable")

# Parse the signals JSON column → individual binary columns
signal_names = [
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

def parse_signals(raw):
    zeros = {f"s_{s}": 0 for s in signal_names}
    # handle None / pd.NA
    if raw is None:
        return zeros
    try:
        if pd.isna(raw):
            return zeros
    except (TypeError, ValueError):
        pass
    # coerce anything non-native (Arrow scalars, etc.) to a plain string first
    if not isinstance(raw, (str, dict, list)):
        raw = str(raw)
    # parse JSON string
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return zeros
    # dict: {"signal_name": 0/1, ...}
    if isinstance(raw, dict):
        return {f"s_{s}": int(raw.get(s, 0)) for s in signal_names}
    # list: either [] or ["signal_name", ...] (present-signal notation)
    if isinstance(raw, list):
        if not raw:
            return zeros
        if all(isinstance(x, str) for x in raw):
            return {f"s_{s}": int(s in raw) for s in signal_names}
        return zeros
    return zeros

signal_df = df["signals"].apply(parse_signals).apply(pd.Series)
df = pd.concat([df.drop(columns=["signals"]), signal_df], axis=1)

# Fill remaining nulls
for col in ["primary_signal", "escalation_stage", "register",
            "language", "persona_id", "category", "hardness_track"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)

# ── Stratified split (by label) ────────────────────────────────────────────────
rng = random.Random(SEED)

pos_idx = df.index[df["label"] == 1].tolist()
neg_idx = df.index[df["label"] == 0].tolist()

rng.shuffle(pos_idx)
rng.shuffle(neg_idx)

def split_indices(idx, train_r, val_r):
    n = len(idx)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

pos_train, pos_val, pos_test = split_indices(pos_idx, TRAIN_RATIO, VAL_RATIO)
neg_train, neg_val, neg_test = split_indices(neg_idx, TRAIN_RATIO, VAL_RATIO)

train_df = df.loc[pos_train + neg_train].sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df   = df.loc[pos_val   + neg_val  ].sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df  = df.loc[pos_test  + neg_test ].sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Train: {len(train_df)} rows  |  Val: {len(val_df)} rows  |  Test: {len(test_df)} rows")
print(f"Train label dist: {train_df['label'].value_counts().to_dict()}")

# ── HuggingFace Features schema ────────────────────────────────────────────────
features = Features({
    "text":             Value("string"),
    "label":            Value("int32"),
    "source":           Value("string"),
    "primary_signal":   Value("string"),
    "escalation_stage": Value("string"),
    "register":         Value("string"),
    "language":         Value("string"),
    "persona_id":       Value("string"),
    "category":         Value("string"),
    # 9 individual signal flags
    "s_burden_language":    Value("int32"),
    "s_finality_language":  Value("int32"),
    "s_escape_framing":     Value("int32"),
    "s_hopelessness":       Value("int32"),
    "s_active_self_harm":   Value("int32"),
    "s_immediate_safety":   Value("int32"),
    "s_self_image_crisis":  Value("int32"),
    "s_third_party_concern":Value("int32"),
    "s_testing":            Value("int32"),
})

# Keep only columns present in features
feature_cols = list(features.keys())
train_df = train_df[[c for c in feature_cols if c in train_df.columns]]
val_df   = val_df  [[c for c in feature_cols if c in val_df.columns]]
test_df  = test_df [[c for c in feature_cols if c in test_df.columns]]

# Cast int columns
for col in feature_cols:
    if features[col].dtype == "int32":
        for split in [train_df, val_df, test_df]:
            split[col] = split[col].astype(int)

# ── Build DatasetDict & push ───────────────────────────────────────────────────
dataset_dict = DatasetDict({
    "train":      Dataset.from_pandas(train_df, features=features),
    "validation": Dataset.from_pandas(val_df,   features=features),
    "test":       Dataset.from_pandas(test_df,  features=features),
})

print(dataset_dict)

dataset_dict.push_to_hub(
    repo_id=HF_REPO_ID,
    max_shard_size="500MB",
    commit_message="Initial release — KHP Youth Mental Health Safety Guardrail Dataset v1.0",
)

print(f"\nDataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")
