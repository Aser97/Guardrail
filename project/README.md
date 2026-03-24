# Project Code

This directory contains the runnable project code — guardrails, providers, notebooks, and evaluation scripts.

> **First time here?** Start with the [Quickstart Guide](../docs/quickstart_guide.md). It walks you through environment setup, your first evaluation run, and building your guardrail step by step.
>
> **Access note:** All required credentials and access information for the BUZZ environment are shared with participants via Slack or email.

<br>

---

<br>

## What's in This Directory

| Path | Description |
|------|-------------|
| **`src/guardrails/`** | Guardrail framework and built-in implementations (LLM judge, classifier). See [src/guardrails/README.md](src/guardrails/README.md). |
| **`src/guardrails/tests/`** | Unit tests for guardrail metrics and prediction flow. |
| **`src/submission/`** | Submission module and example submissions. |
| **`src/prompt_templates/`** | Guardrail prompt templates. |
| **`providers/`** | LLM providers (Cohere, OpenAI, Mistral, demo). |
| **`notebooks/`** | Exploration, guardrail testing, and mmBERT training notebooks. |
| **`scripts/`** | Evaluation workflow scripts (`configure.sh`, `predict.sh`, `evaluate.sh`, `publish_artifact.sh`). |

<br>

---

<br>

## Key Notebooks

| Notebook | Purpose |
|----------|---------|
| `explore_khp_virtual_assistant_cohere.ipynb` | Run adversarial tests against the KHP VA baseline. Use with the **"Python (aiss)"** kernel. |
| `guardrail_evaluation.ipynb` | Run your guardrail on validation data. Shows per-sample predictions, confusion matrix, false positives/negatives, and latency. Your main iteration tool. |
| `train_mmbert_guardrail.ipynb` | Train a fine-tunable classifier guardrail (mmBERT). |

> The [Quickstart Guide](../docs/quickstart_guide.md) covers kernel registration and notebook setup in Steps 10–12.

<br>

---

<br>

## Evaluation Pipeline Scripts

All scripts run from the **repository root**. The pipeline is always **configure -> predict -> evaluate**.

| Script | What it does |
|--------|-------------|
| `scripts/configure.sh` | Creates `.venv`, installs dependencies, validates `hackathon.json`, downloads artifacts. |
| `scripts/predict.sh <input.csv> <output.csv>` | Runs your guardrail on the input dataset and writes a predictions CSV. |
| `scripts/evaluate.sh <predictions.csv> <metrics.csv>` | Computes precision, recall, F1, and latency from predictions. |

```bash
./project/scripts/configure.sh
./project/scripts/predict.sh datasets/seed_validation_set.csv results/predictions.csv
./project/scripts/evaluate.sh results/predictions.csv results/eval_metrics.csv
```

> **Important:** `configure.sh` creates the venv and uses `.venv/bin/python` directly. For `predict.sh`, `evaluate.sh`, and any `python` commands you run yourself, activate the venv first: `source .venv/bin/activate`. See [Quickstart Guide — Step 6](../docs/quickstart_guide.md#step-6--activate-the-virtual-environment).

<br>

---

<br>

## Publishing Model Artifacts

If your submission depends on model files that must be downloaded at runtime (fine-tuned classifier approach), use `publish_artifact.sh` to upload them to S3.

<br>

**Usage** (from the repository root):

```bash
./project/scripts/publish_artifact.sh <team_id> <local_path>
```

- `<team_id>`: your team ID (e.g. `team_001`)
- `<local_path>`: file or directory on your machine

If `<local_path>` is a directory, the script compresses it to `.tar.gz` before uploading. After upload, it prints an artifact block — copy this into `hackathon.json`.

<br>

**Example:**

```bash
./project/scripts/publish_artifact.sh team_001 project/models/mmbert
```

**Output** (paste into `hackathon.json` under `artifacts`):

```
{
  "uri": "s3://hackathon-s3-bucket-999-e8b3s/team_001/mmbert.tar.gz",
  "destination": "ENTER_DESTINATION_PATH_HERE (e.g., project/models)",
  "sha256": "<computed-hash>",
  "required": true
}
```

<br>

**Required environment variables** (set in your `.env` file):

| Variable | Description |
|----------|-------------|
| `S3_BUCKET_NAME` | S3 bucket name (default: `hackathon-s3-bucket-999-e8b3s`) |
| `S3_ENDPOINT_URL` | S3 endpoint URL |
| `S3_ACCESS_KEY` | S3 access key |
| `S3_SECRET_KEY` | S3 secret key |

Without these, `publish_artifact.sh` will fail. See [Quickstart Guide — Step 4](../docs/quickstart_guide.md#step-4--set-up-environment-variables) for `.env` setup.

<br>

---

<br>

## Documentation

| Guide | What it covers |
|-------|---------------|
| [Quickstart Guide](../docs/quickstart_guide.md) | **Start here.** Environment setup through first dashboard score. |
| [Evaluation Dashboard Manual](../docs/evaluation_dashboard_manual.md) | Dashboard UI, triggering evals, reading results, leaderboard. |
| [Guardrails README](src/guardrails/README.md) | Guardrail framework API, built-in implementations, stacking. |
| [Data Generation Manual](../docs/data_generation_manual.md) | Creating training data, taxonomy, DEI guidance. |
| [Red-Team Playbook](../docs/red_team_playbook.md) | Adversarial testing strategies for the KHP VA baseline. |
| [Evaluation & Judging](../docs/evaluation_and_judging.md) | Scoring criteria, rubric, report template. |
| [Policy & Ethics Manual](../docs/policy_ethics_manual.md) | Responsible AI guidance for youth mental health contexts. |
| [Mental Health Safety Primer](../docs/mental_health_safety_primer.md) | Key mental health safety concepts and risk signals. |
| [Docs Index](../docs/README.md) | Full documentation index. |
