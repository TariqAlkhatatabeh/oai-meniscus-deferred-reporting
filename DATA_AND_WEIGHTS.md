# Data and weights

Nothing in this file is distributed with the repository.

## OAI

The Osteoarthritis Initiative baseline knee MRIs are available through the
NIMH Data Archive (doi:10.15154/64dx-3822) under the OAI access and data-use
terms. The study used the 0-month SAG 3D DESS, COR MPR and AX MPR series.

Expected local layout after preprocessing:

```
data/
  kMRI_SQ_MOAKS_BICL00.txt
  OAI_MRI/00m/
  labels.csv
  mri_manifest.csv
  series_catalog.csv
  splits.csv
  processed_16s_v2/
  processed_index_16s_v2.csv
  mrnet_primary_16s_v1_index.csv
  mrnet_da_splits.csv
```

`build_labels.py` derives the binary tear and severity labels from the MOAKS
export. The fixed participant split file (`splits.csv`) is part of the study
record and is not redistributed; it is required by the training and evaluation
loaders.

## MRNet

MRNet is available through the Stanford AIMI research-use distribution. The
16-slice model-facing index is built with:

```bash
python src/preprocess_mrnet.py \
  --root /path/to/MRNet-v1.0 \
  --split train \
  --n_slices 16 \
  --frac 0.80 \
  --tag mrnet_primary_16s_v1
```

The adaptation split file (`data/mrnet_da_splits.csv`) is not redistributed.

## Model assets

Place the following outside the repository:

- `convnext_base.fb_in22k_ft_in1k.safetensors` (ImageNet-pretrained backbone)
- `exp_multitask_16s_v2_convnext_base_safe_strong_bs4_ckpt_20260628_best.pt`
  (primary model checkpoint)
- `Qwen3.5-9B` (text-only reporting model; the reported revision is
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`)
- `Qwen2.5-VL-7B-Instruct` (vision-language model)

Set `MENISCUS_MODELS`, `MENISCUS_WEIGHTS`, `QWEN_TEXT_MODEL`,
`QWEN_VL_MODEL` and `PRETRAINED_CONVNEXT` in the environment, or edit
`config.py`.

Weights, source images and row-level data are excluded because their
redistribution is not covered by the repository licence or the dataset terms.
