# Script map

| Stage | Script | Final output |
|---|---|---|
| Labels | `src/build_labels.py` | `data/labels.csv` |
| OAI preprocessing | `src/preprocess_oai.py` → `src/extract_slices.py` | `data/processed_16s_v2/`, `processed_index_16s_v2.csv` |
| Primary training | `src/train_multitask_16s_v2_timm_exp.py` | checkpoint and validation metrics |
| Primary TTA | `src/eval_multitask_16s_v2_timm_exp_tta.py` | test probabilities |
| Routing/coverage | `src/eval_convnext_confidence_stack.py` | `results/metrics/convnext_safe_strong_16s_tta_20260629_coverage.*` |
| Calibration | `src/eval_primary_model_calibration.py` | calibration metrics |
| Conformal | `src/eval_primary_model_conformal.py` | conformal metrics |
| Bootstrap | `src/bootstrap_performance_ci.py`, `src/bootstrap_extended_ci.py` | intervals and paired differences |
| Grad-CAM | `src/eval_convnext_gradcam_curated.py`, `src/eval_primary_model_gradcam.py` | summary and review figures |
| MRNet preprocessing | `src/preprocess_mrnet.py` | MRNet 16-slice index |
| MRNet adaptation | `src/train_mrnet_primary_16s_da.py` | adapted checkpoint |
| MRNet evaluation | `src/eval_mrnet_primary_16s.py`, `src/compare_mrnet_primary_16s.py` | frozen/adapted and paired metrics |
| Zero-shot VLM | `src/eval_vlm_meniscus.py`, `src/analyze_vlm_eval.py` | packing comparison |
| QLoRA VLM | `src/finetune_vlm_48_strong.py`, `src/analyze_vlm_finetune_48.py` | adapted VLM metrics |
| Fusion | `scripts/fusion/*.py` | validation selection and locked-test aggregates |
| Severity | `scripts/reporting/extract_primary_severity.py` | severity predictions |
| Reporting | `scripts/reporting/generate_reports.py` | paired reports |
| Audit | `scripts/reporting/audit_reports.py` | five-domain aggregates |
| Workload | `scripts/compute_deferral_workload.py` | projected routing workload |

The model, dataset and evaluation scripts use the 16-slice, three-plane
representation. Internal experiment tags remain in artifact filenames so that
the released aggregates can be traced to the files named in the manuscript.
