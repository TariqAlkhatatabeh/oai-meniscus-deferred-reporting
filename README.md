# Calibrated deferral and audited report drafting for knee MRI meniscal tears

Code and aggregate results for the manuscript. The repository covers the
16-slice tri-plane ConvNeXt, validation-locked routing, calibration and
conformal analyses, MRNet transfer and adaptation, the Qwen2.5-VL baseline and
QLoRA adaptation, the ConvNeXt–VLM fusion, and the constrained-reporting and
five-domain audit experiment.

This is the code release only. It is not a clinical system.

## What is here

| Path | Contents |
|---|---|
| `src/` | OAI preprocessing, primary model training/evaluation, calibration and routing, Grad-CAM, MRNet transfer/adaptation, VLM evaluation/adaptation |
| `scripts/reporting/` | Severity extraction, report generation and the five-domain audit |
| `scripts/fusion/` | Validation selection and locked-test evaluation for the ConvNeXt–VLM fusion |
| `scripts/compute_deferral_workload.py` | Projected routing workload |
| `results/metrics/` | Final aggregate metric tables and bootstrap outputs |
| `results/reporting/` | Final report-audit aggregates and the frozen protocol |
| `results/fusion/` | Final fusion aggregates and inference outputs |
| `results/source_data/` | Figure source data |
| `results/vlm_eval/` | Zero-shot VLM summary |
| `results/cohort/` | Aggregate cohort tables |
| `docs/CLAIM_TO_ARTIFACT.tsv` | Claim-to-file map |
| `docs/SCRIPTS.md` | Script-to-stage map |
| `docs/REPRODUCTION.md` | Run order and environment |
| `docs/REPRODUCIBILITY_LIMITATION.md` | What is withheld and why |

## Not included

- OAI and MRNet source images.
- Trained weights and pretrained backbone files.
- Row-level per-knee predictions, per-report audit tables, generated reports,
  per-exam MRNet scores and Grad-CAM case selections.
- The fixed participant split file (`data/splits.csv`).

These are withheld under the OAI and MRNet data-use terms. See
`DATA_AND_WEIGHTS.md` and `docs/REPRODUCIBILITY_LIMITATION.md`.

## Environment

Python 3.11, PyTorch 2.12.0 with CUDA 13.2, timm 1.0.27, Transformers 5.9.0,
Accelerate 1.13.0 and bitsandbytes 0.49.2. Primary training and evaluation used
one RTX 5080 (16 GB). The QLoRA adaptation used 4-bit NF4 quantization with
bfloat16 computation.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the data and model assets as described in `DATA_AND_WEIGHTS.md`, then set
the environment variables in `config.py` as needed:

```bash
export MENISCUS_DATA=/path/to/data
export OAI_MRI_ROOT=/path/to/OAI/00m
export MRNET_ROOT=/path/to/MRNet-v1.0
export QWEN_TEXT_MODEL=/path/to/Qwen3.5-9B
export QWEN_VL_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export PRETRAINED_CONVNEXT=/path/to/convnext_base.safetensors
export MENISCUS_WEIGHTS=/path/to/checkpoints
```

Run from the repository root so `config.py` and the `src` modules resolve.

## Quick start

The full stage order is in `docs/REPRODUCTION.md`. The primary path starts with:

```bash
python src/build_labels.py
python src/preprocess_oai.py --workers 8
python src/train_multitask_16s_v2_timm_exp.py --help
python src/eval_multitask_16s_v2_timm_exp_tta.py --help
python src/eval_convnext_confidence_stack.py --help
python src/eval_primary_model_calibration.py --help
python src/eval_primary_model_conformal.py --help
```

All scripts expose their arguments through `--help`. Paths that depend on the
study data or weights are supplied through `config.py` or the script arguments.

## Results

Aggregate results are in `results/`. `docs/CLAIM_TO_ARTIFACT.tsv` maps each
headline claim to its final aggregate file. Counts that require per-knee rows
are listed as withheld in `docs/REPRODUCIBILITY_LIMITATION.md`.

## Licence and use

MIT for the code only. No right is granted to any OAI or MRNet image, model
weight or withheld row-level table. The workflow was not deployed, and no
reader study or clinical-outcome endpoint was evaluated. Do not use this code
for clinical care.
