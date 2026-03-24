# Quickstart Guide

This guide walks you through every step — from opening your compute environment to synthetizing, training and evaluating your Guard Rail model. Each step builds on the previous one.

---

<br>

## Setup the Environment

Edit the hidden `.env` file with your API keys. You can do so with the CLI command `nano .env` (or another terminal editor). 

The `.env` file is git-ignored and should never be committed. 

### Step 1 — Run configure to build your environment

From the **repository root**, run:

```bash
./project/scripts/configure.sh
```

`configure.sh` does everything for you:

1. Creates a `.venv` virtual environment (if one doesn't exist)
2. Installs all dependencies from `pyproject.toml` (or `requirements.txt` if present)
3. Installs `ipykernel` for notebook support

You should see `OK` at the end. 

### Step 2 — Activate the virtual environment

`configure.sh` created the venv, but you need to activate it in your terminal so that `predict.sh`, `evaluate.sh`, and any Python commands use the right packages:

```bash
source .venv/bin/activate
```

> **Important:** You'll need to run this every time you open a new terminal in JupyterLab. The `configure.sh` script doesn't need activation (it uses `.venv/bin/python` directly), but `predict.sh`, `evaluate.sh`, and any `python` commands you run yourself do.

<br>
---

## Data generation

Run the following scripts to generate the synthetic data using our generation techniques.
All scripts support `--test` (small sample, verbose) and `--append` (resume an interrupted run).
Back up any existing CSV before running a new pass on it.

### Prerequisites
```bash
bash scripts/run_setup.sh
```

---

### Step 3 — Primary generation (run in parallel)

The three scripts below are fully independent and can run simultaneously in separate terminals.

**Direct prompt generation** — instructs the LLM to write a realistic conversation from scratch
for each signal × escalation stage × register × language combination.
French conversations are always routed to Mistral; English/mixed rotate round-robin between
Mistral and Llama-3.3-70B (Together AI).
```bash
bash project/scripts/run_1a.sh             # 540 high-risk + 200 low-risk rows
bash project/scripts/run_1a.sh --test      # 2 per signal + 3 low-risk (smoke test)
```

**CAMEL dual-agent generation — standard** — two LLMs role-play as user in distress and AI support
assistant, generating naturalistic multi-turn conversations. Mistral voices the user; Llama-3.3-70B
voices the assistant. Each conversation is accepted only if a Claude judge scores it above the
realism and signal-presence thresholds.
```bash
bash project/scripts/run_1b.sh             # 300 conversations
bash project/scripts/run_1b.sh --test      # 3 conversations (smoke test)
```

**CAMEL dual-agent generation — hard constraints** — same dual-agent setup but with two enforced
difficulty tracks: *subtle* (signal always expressed indirectly, subtlety ≥ 7) and *escalating*
(conversation arc starts low-risk and builds to a signal peak, escalation arc ≥ 7). Uses a
chain-of-thought Claude judge that reasons step-by-step before scoring.
```bash
bash project/scripts/run_1b_hard.sh        # 300 conversations, 50/50 split across tracks
bash project/scripts/run_1b_hard.sh --test # 4 conversations (smoke test)
```

---

### Step 4 — Augmentation and adversarial generation (run in parallel after Step 1)

**Seed augmentation** — applies four transformation operators to the curated seed validation set:
language rewrite (EN → FR/ES/PT), persona swap (age/register/background), signal injection
(low-risk → borderline/high-risk), and signal softening (high-risk → borderline). Requires
`datasets/seed_validation_set.csv` (committed to the repo).
```bash
bash project/scripts/run_2abcd.sh          # ~80 rows per transformation type
bash project/scripts/run_2abcd.sh --test   # 2 per type (smoke test)
```

**PAIR adversarial generation — standard** — a generator–judge red-teaming loop: Mistral generates
a conversation, Claude judges it and returns structured feedback, Mistral revises, repeat up to
`--max_iter` times. Produces two row types: *hard positives* (signal present but expressed
indirectly) and *adversarial negatives* (surface language mimics high-risk but content is
definitively low-risk).
```bash
bash project/scripts/run_2g.sh             # 200 conversations
bash project/scripts/run_2g.sh --test      # 4 conversations (smoke test)
```

**PAIR adversarial generation — hard constraints** — same generator–judge loop but with three
enforced tracks (subtle, escalating, adversarial negative) and stricter scoring gates. The Claude
judge uses explicit chain-of-thought reasoning before each accept/reject decision.
```bash
bash project/scripts/run_2g_hard.sh        # 200 conversations
bash project/scripts/run_2g_hard.sh --test # 6 conversations (smoke test)
```

---

### Step 5 — Quality control

See [Quality_control.md](Quality_control.md) for the recommended
post-generation quality passes (realism filtering, signal-survival checks, artifact removal).

---

### Step 6 — Assembly
```bash
bash project/scripts/run_assemble.sh       # merges all CSVs into datasets/final_dataset.csv
```

---

## Training the Guard Rail

### Step 7 — Train the guardrail model

Trains Qwen2.5-7B-Instruct with LoRA (r=16, α=32) as a 9-head multi-label classifier
using BCEWithLogitsLoss. Requires a GPU (tested on A6000 48 GB, bf16, ~3–4 h).
No API keys needed — training is fully local.

Must run after Step 4 (`run_assemble.sh` has produced `datasets/train.csv`).
```bash
bash project/scripts/run_train.sh             # full run (3 epochs)
bash project/scripts/run_train.sh --test      # sanity check (1 epoch, small batch)
bash project/scripts/run_train.sh --resume    # resume from checkpoint_last/ after a crash
```

Artifacts are written to `project/models/mhs_guardrail/`.

---

### Step 8 — Calibrate the decision threshold

Reconstructs the exact validation split used during training (seed=42, 15% hold-out),
sweeps thresholds from 0 → 1 across the LR aggregation head outputs, and prints two
recommendations: the Youden's J maximiser (balances sensitivity and specificity) and
the recall ≥ 95% threshold (preferred for safety-critical deployment, minimises false
negatives).
```bash
python project/scripts/calibrate_threshold.py \
    --data          datasets/train.csv \
    --model_dir     project/models/mhs_guardrail \
    --device        cuda \
    [--batch_size   4] \
    [--output       calibration_results.json]
```

The script prints the exact value to paste into `submission.py`:
```python
_THRESHOLD = <recommended value>   # update this line in src/submission/submission.py
```

---


## Evaluate our GuardRail

### Step 9 — Run predict

```bash
./project/scripts/predict.sh datasets/seed_validation_set.csv results/predictions.csv
```

This loads the guardrail from `submission.py`, runs it on every row in the validation dataset, and writes a predictions CSV.

<br>

### Step 10 — Run evaluate

```bash
./project/scripts/evaluate.sh results/predictions.csv results/eval_metrics.csv
```

This computes precision, recall, F1, and latency from your predictions. Open `results/eval_metrics.csv` or `results/eval_metrics.json` to see the numbers.

<br>

### Step 11 — Register the notebook kernel

`configure.sh` already installed `ipykernel`, so you just need to register it. From the repo root with your venv activated:

```bash
python -m ipykernel install --user --name=aiss --display-name="Python (aiss)"
```

> **Important:** After registering, you must select the correct as the kernel every time you open a notebook in JupyterLab. The environment may default to a different kernel (e.g. the base Python kernel), which will not have dependencies installed. To change the kernel: click **Kernel > Change Kernel** in the menu bar and select. If you see `ModuleNotFoundError` when running notebook cells, this is almost always a wrong-kernel issue.

<br>

### Step 12 — Explore results in the evaluation notebook

Open `project/notebooks/guardrail_evaluation.ipynb` in JupyterLab and select the **"Python (aiss)"** kernel.

This notebook goes deeper than the command-line scripts. It loads a submission module, runs the guardrail on the validation data, and gives you:

- **Aggregate metrics** — precision, recall, F1, latency
- **Full predictions table** — every sample with its ground-truth label, your guardrail's prediction, and per-sample latency
- **Confusion matrix** — TP, FP, TN, FN counts at a glance
- **False positive analysis** — low-risk content your guardrail incorrectly flagged (false alarms)
- **False negative analysis** — high-risk content your guardrail missed (the dangerous ones)
- **Latency distribution** — how inference time varies across samples
