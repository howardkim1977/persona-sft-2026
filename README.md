# Reproducibility package

**Effects of Grounded Synthetic Persona Conditioning on Fine-Tuning a
Small Korean Language Model: A Preregistered Controlled Comparison Against
No-Persona and Ad-Hoc Persona Data**

Preregistration: https://osf.io/fc8mn (registered 2026-08-04, before any
main-experiment data collection).

## What this package contains

| Path | Contents |
|---|---|
| `code/` | Generation, evaluation, judging, and training-prep scripts; design notes (DESIGN.md) |
| `analysis/` | Confirmatory and exploratory analysis scripts and their outputs (JSON/TXT) |
| `data/` | 30,665 generation records with per-item metadata; the three persona partitions (train / reserve / evaluation) |
| `eval/` | Evaluation sets (H1 500 prompts, H2 400 items, H3 6,561 items) and all 21 response files (7 models x 3 parts) |
| `judge/` | Raw scoring log, 11,200 judgements (4 judges x 400 items x 7 models) |
| `adapters/` | Six LoRA adapters (selected lowest-validation-loss checkpoints) |
| `paper/` | Manuscript source, figure scripts, figures, similarity self-check |
| `licenses/`, `docs/` | Model licenses, terms-of-service verification record, preregistration draft |
| `MANIFEST.sha256` | SHA-256 of every file |

Not included: the NVIDIA Nemotron-Personas-Korea source CSV (4.1 GB;
publicly distributed by NVIDIA) and API keys. `data/` ships the exact
persona partitions drawn from it, so the pipeline is reproducible without
re-downloading the source.

## Reproducing the analyses

The confirmatory results can be recomputed from the shipped artifacts
without any API calls:

```bash
python3 analysis/manip_check.py        # Table-01
python3 analysis/h1_analysis.py        # Table-02 (bootstrap 10,000)
python3 analysis/h2_analysis.py        # Table-03, primary hypothesis
python3 analysis/exploratory.py        # Figure-02 inputs, judge stability
python3 analysis/verify_manuscript.py  # checks every number in the paper
```

`verify_manuscript.py` cross-checks 91 reported values against these
outputs and exits with a list of any mismatches.

Regenerating data or responses requires API access (OpenRouter, CLOVA
Studio) and the environment variables documented in `code/CLAUDE.md`.

## Adapters

Adapters are LoRA derivatives of `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`
and are distributed under the EXAONE AI Model License Agreement 1.1-NC:
names begin with "EXAONE", each directory carries a copy of the license
and a NOTICE, and use is restricted to research / non-commercial purposes.

Load with mlx_lm 0.31.3:

```bash
python3 -m mlx_lm generate \
  --model mlx-community/EXAONE-3.5-2.4B-Instruct-bf16 \
  --adapter-path adapters/EXAONE-3.5-2.4B-personaSFT-C2-seed42-adapter \
  --prompt "..." --temp 0.0
```

## Known issues, disclosed

1. Persona identifiers were absent from the original per-item records
   because the source CSV carries a UTF-8 byte-order mark; they were
   restored post hoc from deterministic selection indices
   (`analysis/repair_uuid.py`). Generation records are fully restored;
   65 of 400 H2 items remain unattributed because several evaluation-pool
   personas share identical demographic segments. The train/evaluation
   split is disjoint by construction and verified (intersection 0).
2. Judge API usage was not logged per call, so cost figures for scoring
   are token-count estimates rather than metered values.
3. Response temperature for H2 (0.0) was an implementation choice made
   after registration; see Section 3.3 of the manuscript.
