# AIML335 Assignment 1: Chronicles Across the Ages

## Overview

This submission implements both parts of the assignment inside the provided starter
notebook, `assignment1_student_template.ipynb`:

- **Part 1**: a class-incremental continual-learning router (naive sequential training,
  full replay, bounded replay at two buffer sizes), evaluated for catastrophic forgetting
  across three Experiences.
- **Part 2**: a cosine-similarity retriever over MiniLM embeddings (era-restricted vs.
  full-corpus, whole-passage vs. chunked), plus a small local generation model comparing
  parametric vs. retrieved knowledge on 12 matched real/fantasy question pairs.

## Requirements

- Python 3.10+
- Packages: `numpy`, `pandas`, `scikit-learn`, `openai` (used as an HTTP client against
  the local LM Studio server, not OpenAI's hosted API)
- [LM Studio](https://lmstudio.ai/) running locally with two models loaded:
  - An embedding model: `text-embedding-all-minilm-l6-v2-embedding` (any quantization,
    e.g. `@q4_k_s`)
  - A chat model: `llama-3.2-3b-instruct` (or another edge-class instruct model per the
    assignment brief, e.g. Qwen2.5 1.5B Instruct)
- The data files `passages.json`, `questions.json`, `baseline_questions.json` in the same
  folder as the notebook (or update `DATA_DIR` in the setup cell)

Install dependencies:
```
pip install numpy pandas scikit-learn openai
```

## Setup

1. Start LM Studio and load both models above. Confirm they're being served:
   ```
   curl http://localhost:1234/v1/models
   ```
2. Open `assignment1_student_template.ipynb` and edit the setup cell near the top with
   your own model IDs, exactly as reported by the command above:
   ```python
   EMBED_MODEL_ID = "text-embedding-all-minilm-l6-v2-embedding@q4_k_s"
   CHAT_MODEL_ID = "llama-3.2-3b-instruct"
   ```
3. Confirm `DATA_DIR` points at the folder containing the three JSON data files.

## Running

Restart the kernel and run all cells top to bottom. Part 1 does not require the chat
model, only the embedding model. Part 2(c) requires both, and involves 36 local LLM calls
(3 conditions x 12 question pairs), so it is the slowest cell to run.

A fixed random seed (`SEED = 335`) is used throughout for reproducibility, in the
stratified train/test split, the replay buffer sampling, the MLP's weight
initialization, and the part 2(c) pair selection.

## Notes on design choices

- **Part 1 replay buffer sizes**: bounded replay was tested at 3 and 5 passages per
  earlier era (within the assignment's suggested 3 to 5 range).
- **Seed averaging**: Part 1's results are averaged over 5 seeds (335 to 339) using
  `average_over_seeds()` in the accuracy-matrix cell, per the notebook's optional
  "recommended" TODO, to reduce noise from any single train/test split or weight
  initialization.
- **Part 2 chunking**: each passage is split into 2 chunks at the nearest sentence
  boundary to its midpoint (not mid-sentence), rather than a fixed word count.
- **Part 2(c) sampling**: 2 matched real/fantasy pairs were sampled per era (12 total),
  chosen with `SEED` for reproducibility.

## AI tool use disclosure

Significant AI assistance (Claude) was used throughout this assignment for implementation
of the graded TODOs, interpretation of results, and drafting of the written discussions,
per written permission from the course lecturer (email on file, dated 24 July 2026)
extending the original AI-use policy to allow this, conditional on the student being able
to explain all code and results independently if asked.

## Submission contents

- `assignment1_student_template.ipynb`: completed notebook (code, result tables/figures,
  and written discussion for both parts)
- `README.md`: this file
