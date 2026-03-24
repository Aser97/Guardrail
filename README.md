# Hackathon: Building Safer AI for Youth Mental Health

Welcome to the Building Safer AI for Youth Mental Health Hackathon. This repository contains starter code, an evaluation framework, datasets, and documentation to build and submit safety guardrails.

![AI Guardrail Challenge overview](hackathon_overall.png)
*Participants are expected to red-team the KHP bot through adversarial testing and, in blue-teaming, build a strong input guardrail that reduces harmful misses while preserving safe usability.*

<br>

> **New here?** Start with the [Quickstart Guide](docs/quickstart_guide.md) — it walks you from opening your compute environment to getting your first evaluation score on the dashboard.
>
> **Access note:** Participants will have access to the BUZZ environment with A40 GPU support, where they can set up their code and train models. Required credentials and access details (including S3 buckets and chat login information) will be shared via Slack or email.
>
> **Chatbot purpose:** The KHP chatbot app is a **stress-testing and data-gathering tool only**. Use it to explore how the virtual assistant handles realistic scenarios, identify where it fails, and collect examples you can use as training data. It does **not** train your model automatically. After gathering findings from the chatbot, you must manually retrain/update your model, add observations to your report, then commit and push your code before triggering the evaluation dashboard.
>
> **Participant well-being:** This hackathon involves working with sensitive mental health content. If at any point during the event you feel stressed or overwhelmed, please reach out to [Kids Help Phone](https://kidshelpphone.ca/) — they are here to support you.

<br>

---

<br>

## Background

The hackathon has two parts around the KHP (Kids Help Phone) virtual assistant:

- **Part 1: Stress testing** the KHP chatbot to uncover realistic failure modes and gather training data.
- **Part 2: Blue-team hardening** by building a strong input guardrail that improves 
`low_risk` vs `high_risk` detection.

<br>

---

<br>

## What You Must Submit

Both a **report** and **code** are required. Submissions missing either component will not be evaluated.

| Deliverable | What's expected |
|-------------|----------------|
| **One-pager + full report** (`docs/report.pdf`) | Adversarial testing findings, methodology, data generation details, dataset statistics, performance results. Follow the structure in [evaluation_and_judging.md](docs/evaluation_and_judging.md). |
| **Code** (`project/src/submission/submission.py`) | Your guardrail logic via `get_guardrails()` returning `(input_guardrail, None)`. |
| **Model artifacts** (if fine-tuned) | Uploaded to S3 and declared in `hackathon.json`. |

See the full [Submission Checklist](#submission-checklist) below for details and validation steps.

<br>

---

<br>

## Hackathon Task

This challenge is intentionally split into two exercises:

1. **Part 1 — Stress testing the chatbot (not security penetration testing)**
   Use `project/notebooks/explore_khp_virtual_assistant_cohere.ipynb` and the chatbot app [https://chatbot-app.hackathon.buzzperformancecloud.com/](https://chatbot-app.hackathon.buzzperformancecloud.com/) to stress test realistic youth-support scenarios. The chatbot is your tool for **exploring vulnerabilities** and **generating training data**. Focus on cases where vulnerable youth could receive harmful information, unsafe guidance, or insufficient escalation to human support.
   > **Scope note:** Stress testing in this hackathon targets realistic edge cases where the chatbot may miss risk signals, under-escalate unsafe situations, or unintentionally reinforce harmful choices. Think about what a real user in distress might encounter — do **not** focus on malicious prompt injection or hacker-style exploitation.
   >
   > **Data gathering:** The chatbot does **not** train your model. It is solely for exploring the VA's behavior and collecting examples. After using it, manually incorporate your findings into your training data and report.

2. **Document stress-testing findings in the report**
   Include concrete examples of failure modes, risk patterns, and how your approach addresses them. This analysis is a required part of your submitted report. See the [Red-Team Playbook](docs/red_team_playbook.md) for structured strategies.

3. **Part 2 — Blue-teaming (build and harden input guardrails)**
   Update `project/src/submission/submission.py` so `get_guardrails()` returns `(input_guardrail, None)`. Your `submission.py` changes in the latest pushed Git commit are what the automated evaluation pipeline runs.

4. **Improve risk assessment quality**
   Your input guardrail should better classify interactions as `low_risk` vs `high_risk` and reduce harmful misses/false alerts.

<br>

**Supported approaches:**

| Approach | Example file |
|----------|-------------|
| LLM judge guardrail | `example_submission_cohere_llm_judge.py` |
| Fine-tuned model guardrail | `example_submission_mmbert_guardrail.py` |
| Baseline template guardrail | `example_submission.py` |
| Stacked input guardrails | `example_stacked_llm_model.py` |

All example files live in `project/src/submission/`. Use `project/notebooks/guardrail_evaluation.ipynb` to validate guardrail behavior and metrics.

<br>

---

<br>

## What You Change

Out of all the files in this repo, you only need to customize **two**:

| File | What to put in it |
|------|-------------------|
| **`hackathon.json`** | `needs_gpu` (boolean) + `artifacts` (list of model downloads) |
| **`project/src/submission/submission.py`** | Your guardrail logic via `get_guardrails()` |

Everything else — `configure.sh`, `predict.sh`, `evaluate.sh`, the guardrails framework, providers — is shared scaffold. **Do not modify the shared scripts or evaluator-facing interfaces.**

<br>

---

<br>

## Submission Contract

Your submission is a **single Python module** in `project/src/submission/` that defines:

```python
def get_guardrails() -> tuple[input_guardrail, output_guardrail]:
    """Return (input_guardrail, output_guardrail). Either may be None."""
    ...
```

- **No arguments** are passed to `get_guardrails()`.
- For this hackathon, return your **input** guardrail and **`None`** for output.
- Each return value may be `None`, a **single** guardrail, or a **list/tuple** (stack). Stacks run in order.
- You may use your own LLM, classifier/BERT, or any guardrail that implements the guardrail protocol.

For the full guardrail API, built-in implementations, and stacking behavior, see the [Guardrails README](project/src/guardrails/README.md).

<br>

---

<br>

## Repository Structure

| Path | Description |
|------|-------------|
| **`project/`** | Main code: guardrails, providers, notebooks. **Do your development here.** See [project/README.md](project/README.md). |
| **`project/src/submission/`** | Submission module and example submissions. |
| **`project/src/guardrails/`** | Guardrail framework and built-in implementations. See [Guardrails README](project/src/guardrails/README.md). |
| **`project/notebooks/`** | Exploration, guardrail evaluation, and mmBERT training notebooks. |
| **`project/scripts/`** | Evaluation workflow scripts (`configure.sh`, `predict.sh`, `evaluate.sh`, `publish_artifact.sh`). |
| **`datasets/`** | Starter training/validation CSVs. See `datasets/README.md` for the taxonomy and risk signal table. |
| **`docs/`** | All documentation guides. See [docs/README.md](docs/README.md). |

<br>

---

<br>

## Datasets

- `datasets/seed_validation_set.csv` is a **portion of the final evaluation test data** that the organizer team will use to evaluate models.
- Participants are encouraged to **create their own training data** and aim to beat the provided baselines.
- If you create a dataset for fine-tuning, place it in the **`datasets/`** folder so it can be included in evaluation context and review.
- The full taxonomy, risk signal table, and implementation notes are documented in `datasets/README.md`.

For guidance on generating high-quality training data (taxonomy mapping, hard negatives, DEI coverage, bilingual phrasing, multi-turn patterns), see the [Data Generation Manual](docs/data_generation_manual.md).

<br>

---

<br>

## Submission Deadline

**Sunday, March 22 | 20:00 (Virtual)**

This is the final deadline for teams to submit all required materials: source code pushed to the repository `main` branch, a comprehensive report, and a one-page presentation summary. Finalists will be invited to present their work to the honorary judges on March 23.

<br>

---

<br>

## Submission Checklist

<br>

### 1. Report (required)

Submit a **one-pager** (pitch) and a **full report** as `docs/report.pdf`.

Your report should include:

- Adversarial testing findings with annotated examples
- Methodology and guardrail design decisions
- Data generation details (DEI criteria, bilingual coverage, cultural diversity)
- Dataset statistics (total examples, label distribution, language split, diversity slices)
- Performance results on the validation set (F1 + latency)

Follow the required structure in [docs/evaluation_and_judging.md](docs/evaluation_and_judging.md).

<br>

### 2. Code (required)

- Update **only** `project/src/submission/submission.py` for your guardrail logic.
- Keep the submission contract unchanged: `get_guardrails()` returns `(input_guardrail, None)`.
- You may return a single input guardrail or a stack.

<br>

### 3. Do not modify shared interfaces (required)

- Do **not** change `configure.sh`, `predict.sh`, or `evaluate.sh`.
- Do **not** change the submission contract or evaluator-facing interfaces.

<br>

### 4. Model artifacts (required for fine-tuned models)

If your guardrail uses a fine-tuned model, upload it to S3 and declare it in `hackathon.json`. See [project/README.md — Publishing Model Artifacts](project/README.md#publishing-model-artifacts) for instructions.

<br>

### 5. Local validation (required)

Run the official workflow from the repo root before submitting:

```bash
./project/scripts/configure.sh
./project/scripts/predict.sh datasets/seed_validation_set.csv results/predictions.csv
./project/scripts/evaluate.sh results/predictions.csv results/eval_metrics.csv
```

The [Quickstart Guide](docs/quickstart_guide.md) covers this in detail (Steps 5–9 and 17).

<br>

### 6. Push latest code before dashboard trigger (required)

- The evaluation pipeline clones your repository and runs the latest pushed commit at trigger time.
- Unpushed local changes are not evaluated.

<br>

---

<br>

## Evaluation Criteria

| Category | What it measures |
|----------|-----------------|
| **Guardrail quality** | Precision, recall, F1 on the evaluation set — strong detection of high-risk interactions |
| **Dataset quality** | Quality of generated/curated data, diversity coverage, clarity of dataset statistics |
| **Efficiency** | Latency and runtime reliability in the configure/predict/evaluate flow |
| **Generalization** | Robustness to adversarial, noisy, and edge-case phrasing |
| **Method quality** | Clarity of problem analysis, methodology choices, and data strategy (one-pager) |
| **Responsible data practice** | Labeling quality, DEI coverage, multilingual considerations, bias-awareness |

For judging stages, disqualification rules, and the full rubric, see [docs/evaluation_and_judging.md](docs/evaluation_and_judging.md).

<br>

---

<br>

## Finalist Selection

- Top finalists are selected using both parts of the challenge: **automated evaluation results** (blue-team guardrail performance) and **report evaluation** (red-team stress-testing quality and data analysis). See [Evaluation & Judging](docs/evaluation_and_judging.md) for scoring details and rubric criteria.
- Finalists will be invited to deliver a **10-minute presentation** on the final day.

<br>

---

<br>

## AI Usage Policy

### Permitted use

- **General development assistance:** You may use any external model (e.g. GPT-4, Claude, Gemini) for coding support, brainstorming, architecture planning, and synthetic data generation.
- **Guardrail models:** Input guardrails must use models hosted on the BUZZ environment. Supported LLM-judge providers are **Cohere**, **Mistral**, and **OpenAI**. Be aware that LLM-judge approaches add inference latency — **fine-tuned classifiers are recommended** for the best balance of accuracy and speed.

### Not permitted

- Using any external model not hosted on BUZZ as a runtime guardrail (e.g. calling an external API from your `submission.py`).

For full responsible-use guidance, see the [Policy & Ethics Manual](docs/policy_ethics_manual.md).

<br>

---

<br>

## Documentation

| Guide | What it covers |
|-------|---------------|
| **[Quickstart Guide](docs/quickstart_guide.md)** | **Start here.** Environment setup through first dashboard score. |
| [Evaluation Dashboard Manual](docs/evaluation_dashboard_manual.md) | Dashboard UI, triggering evals, reading results, leaderboard. |
| [Guardrails README](project/src/guardrails/README.md) | Guardrail framework API, built-in implementations, stacking. |
| [Data Generation Manual](docs/data_generation_manual.md) | Creating training data, taxonomy, DEI guidance. |
| [Red-Team Playbook](docs/red_team_playbook.md) | Adversarial testing strategies for the KHP VA baseline. |
| [Evaluation & Judging](docs/evaluation_and_judging.md) | Scoring criteria, rubric, report template. |
| [Policy & Ethics Manual](docs/policy_ethics_manual.md) | Responsible AI guidance for youth mental health contexts. |
| [Mental Health Safety Primer](docs/mental_health_safety_primer.md) | Key mental health safety concepts and risk signals. |
| [Project README](project/README.md) | Directory layout, evaluation scripts, publishing artifacts. |
| [Docs Index](docs/README.md) | Full documentation index. |

<br>

---

<br>

## Terms and Conditions

By submitting this solution to the Hackathon, the participant acknowledges and agrees to abide by the Event’s [Terms and Conditions](https://drive.google.com/drive/folders/1-sD05Bcc3oBo2RlyNOEiavhnJUQZHjQp?usp=sharing)
