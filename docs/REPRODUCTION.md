# Reproduction order

Run from the repository root. The scripts use `config.py` for paths and expose
their remaining options through `--help`. The outputs below are aggregate
metrics; scripts that write row-level predictions, reports or audit tables are
marked. Those row-level outputs are not redistributed.

## 1. OAI cohort and images

```bash
python src/build_labels.py
python src/preprocess_oai.py --workers 8
```

`build_labels.py` reads the MOAKS export and writes `data/labels.csv`. The fixed
participant split file is required but is not included. `preprocess_oai.py`
creates the 16-slice, three-plane PNG representation and its index.

## 2. Primary model

```bash
python src/train_multitask_16s_v2_timm_exp.py --help
python src/eval_multitask_16s_v2_timm_exp_tta.py --help
python src/eval_convnext_confidence_stack.py --help
python src/eval_primary_model_calibration.py --help
python src/eval_primary_model_conformal.py --help
python src/bootstrap_performance_ci.py --help
python src/bootstrap_extended_ci.py --help
```

The primary checkpoint was selected on validation AUC at epoch 14. The reported
test evaluation uses seven-view TTA. Temperature and routing cutoffs are fitted
on validation data only. The coverage, calibration, conformal and paired
tests are written under `results/metrics/`.

## 3. Grad-CAM

```bash
python src/eval_convnext_gradcam_curated.py --help
```

This is a qualitative attention review, not lesion segmentation. Raw Grad-CAM
case selections are withheld.

## 4. MRNet transfer and adaptation

```bash
python src/preprocess_mrnet.py --help
python src/train_mrnet_primary_16s_da.py --help
python src/eval_mrnet_primary_16s.py --help
python src/compare_mrnet_primary_16s.py --help
```

The frozen model is evaluated without MRNet fitting. The adaptation run uses a
fixed 874/188/188 examination split; the split file is withheld. The reported
frozen benchmark on the official 120-examination validation subset is a
descriptive estimate and is not independent of the adaptation dataset.

## 5. VLM baseline and adaptation

```bash
python src/eval_vlm_meniscus.py --help
python src/analyze_vlm_eval.py --help
python src/finetune_vlm_48_strong.py --help
python src/analyze_vlm_finetune_48.py --help
```

The zero-shot evaluation uses four input packings. The adaptation uses rank-16
QLoRA on the fixed training partition and selects the adapter on validation
AUC. The combined result is a single locked-test scoring pass.

## 6. Secondary fusion

```bash
python scripts/fusion/select_on_validation.py --help
python scripts/fusion/evaluate_locked_test.py --help
python scripts/fusion/paired_auc_cluster_bootstrap_inference.py --help
python scripts/fusion/paired_auc_cluster_permutation.py --help
```

The 70:30 ConvNeXt–VLM blend is selected on validation only. The locked test is
read once. The fusion analysis is secondary and does not change the primary
pipeline.

## 7. Reporting and audit

```bash
python scripts/reporting/extract_primary_severity.py --help
python scripts/reporting/generate_reports.py --help
python scripts/reporting/audit_reports.py --help
```

The reporting experiment uses only the eight structured primary-model fields.
All 566 test knees are generated once under the constrained prompt and once
under the unconstrained control. The frozen five-domain auditor scores both
arms. Generated reports and the per-report audit table are withheld.

## 8. Routing workload

```bash
python scripts/compute_deferral_workload.py
```

This reproduces the projected 1,000-knee workload from the locked routing
tables; the per-knee routing table is withheld.

## Statistics

All test intervals are 95% participant-clustered bootstrap intervals with 5,000
replicates; all knees from a participant move together. Paired differences use
the same resampling. MRNet intervals resample examinations. All seed values are
reported in the corresponding metrics artifacts; the primary seed is 42.
