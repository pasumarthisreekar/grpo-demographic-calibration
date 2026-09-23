# GRPO Reinforcement Fine-Tuning for Demographic Response Calibration

Fine-tunes Llama-3-8B-Instruct with GRPO (Group Relative Policy Optimization) to align a model's *distribution* of generated survey answers with real-world demographic response patterns, rather than optimizing for a single "correct" answer.

## What it does

- Extracts real US survey response distributions from two public datasets — **GSS** (General Social Survey) and **ANES** (American National Election Studies) — for two demographic personas: Urban High-Income and Rural Working-Class
- Fine-tunes **Llama-3-8B-Instruct** with **GRPO** (via TRL's `GRPOTrainer` + LoRA), using a custom reward function that scores how closely the *distribution* of 8–32 sampled completions matches the real-world ground-truth split for that persona/question
- Reward function uses **Gemini** as an LLM classifier to map free-text completions to discrete stances, with explicit anti-reward-hacking logic to penalize refusals, gibberish, and keyword-spam completions
- Evaluates calibration using **Jensen-Shannon Divergence** between the model's generated response distribution and ground truth, on held-out (out-of-distribution) questions

## Results

Averaged across 10 held-out test questions:

| Metric | Base Model | GRPO Fine-Tuned |
|---|---|---|
| Avg. Jensen-Shannon Divergence | 0.4165 | 0.2491 |

A **40% reduction** in average JSD — meaning the fine-tuned model's answer distribution is substantially closer to real demographic survey response patterns than the base model's.

## Pipeline

```
GSS / ANES (public survey datasets)
      │
      ▼
Persona extraction (quantile-bucket by income + urbanicity)
      │
      ▼
Ground-truth response distributions per persona/question
      │
      ▼
GRPO training (TRL + LoRA)
   reward = density-ratio(target_dist, observed_dist), clipped [-1, 1]
   observed_dist computed via Gemini-based stance classifier
      │
      ▼
Evaluation: Jensen-Shannon Divergence vs. base model, on held-out questions
```

## Tech stack

Python, PyTorch, TRL, LoRA, Transformers, Gemini API, Pandas

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```
3. Place the GSS and ANES source files under `data/` (not included in this repo — see Data Sources below), then generate persona distributions:
   ```bash
   python generate_gss_distributions.py
   python generate_gss_test_distributions.py
   python extract_anes_texts.py
   ```
4. Run GRPO training:
   ```bash
   python train_grpo.py
   ```
5. Evaluate calibration (Jensen-Shannon Divergence) — see `evaluation_metrics.ipynb`.

## Files

- `generate_gss_distributions.py` — extracts GSS survey response distributions by persona for training
- `generate_gss_test_distributions.py` — same, for a held-out set of test questions
- `extract_anes_texts.py` — extracts open-ended ANES response text by persona
- `anes_human_texts.json` — extracted ANES open-ended responses by persona (output of `extract_anes_texts.py`)
- `reward_function.py` — the GRPO reward function (density-ratio scoring + Gemini stance classifier)
- `train_grpo.py` — GRPO training script (TRL + LoRA)
- `train_grpo_notebook.ipynb` — training run notebook
- `evaluation_metrics.ipynb` — computes Jensen-Shannon Divergence between model and ground-truth distributions, with the final results

## Data sources

This project uses two public survey datasets. The raw source files (GSS `.csv`, ANES `.csv`/`.xlsx`) are not redistributed here due to size — download them separately and place them under `data/` before running the pipeline scripts. The derived output (`anes_human_texts.json`) is included in this repo.

- [General Social Survey (GSS)](https://gss.norc.org/)
- [American National Election Studies (ANES) 2020 Time Series](https://electionstudies.org/data-center/2020-time-series-study/)
