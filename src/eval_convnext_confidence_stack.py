"""Evaluate primary-model discrimination, calibration, routing and comparison."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import config as cfg
from dataset_multitask_16s_v2_aug import (
    MeniscusDatasetMT16SV2,
    load_mt_split,
    verify_no_leakage_mt,
)
from eval_multitask_16s_v2_timm_exp_tta import (
    build_tta_pipeline,
    load_model_from_checkpoint,
    predict_with_tta,
)
from eval_primary_model_calibration import (
    apply_temperature,
    build_bin_table,
    fit_temperature,
    metrics_for as calibration_metrics_for,
    plot_calibration_figure,
)
from eval_primary_model_conformal import (
    build_prediction_rows,
    conformal_quantile,
    plot_conformal_figure,
    summarize_split,
    true_label_scores,
)
from routing_utils import (
    coverage_result,
    cutoff_for_target_coverage,
    make_prediction_frame,
    metrics_for_subset,
    plot_coverage,
)
from gradcam_utils import (
    merge_test_rows_with_predictions,
    normalize_cam,
    resize_cam_np,
    select_plausible_example_cases,
    summarize_cam_plane,
    summarize_plausibility,
)


DEFAULT_TAG = "convnext_safe_strong_16s_tta_20260629"
DEFAULT_CKPT = (
    cfg.WEIGHTS /
    "exp_multitask_16s_v2_convnext_base_safe_strong_bs4_ckpt_20260628_best.pt"
)
DEFAULT_RESNET_TTA = (
    cfg.RESULTS / "exp_multitask_16s_v2_aug_safe_strong_20260627_tta_7x_results.json"
)
DEFAULT_RESNET_COVERAGE = (
    cfg.RESULTS / "metrics" /
    "primary_model_safe_strong_16s_tta_coverage_20260628.json"
)
DEFAULT_RESNET_CALIBRATION = (
    cfg.RESULTS / "metrics" /
    "primary_model_safe_strong_16s_tta_calibration_20260628.json"
)
DEFAULT_RESNET_CONFORMAL = (
    cfg.RESULTS / "metrics" /
    "primary_model_safe_strong_16s_tta_conformal_20260628_v2_metrics.csv"
)
DEFAULT_RESNET_GRADCAM = (
    cfg.RESULTS / "metrics" /
    "primary_model_safe_strong_16s_tta_gradcam_20260628_v2_summary.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConvNeXt confidence/calibration/conformal/Grad-CAM stack")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"],
                        default="fp16")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--coverages", type=float, nargs="+",
                        default=[0.90, 0.80, 0.70])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--max_gradcam_per_group", type=int, default=8)
    parser.add_argument("--gradcam_target",
                        choices=["predicted_class", "tear_logit"],
                        default="predicted_class")
    parser.add_argument("--skip_gradcam", action="store_true")
    parser.add_argument("--resnet_tta_json", type=Path,
                        default=DEFAULT_RESNET_TTA)
    parser.add_argument("--resnet_coverage_json", type=Path,
                        default=DEFAULT_RESNET_COVERAGE)
    parser.add_argument("--resnet_calibration_json", type=Path,
                        default=DEFAULT_RESNET_CALIBRATION)
    parser.add_argument("--resnet_conformal_csv", type=Path,
                        default=DEFAULT_RESNET_CONFORMAL)
    parser.add_argument("--resnet_gradcam_summary_csv", type=Path,
                        default=DEFAULT_RESNET_GRADCAM)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def guard_outputs(paths, overwrite: bool):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing ConvNeXt stack outputs. Use a new "
            "--tag or pass --overwrite.\n" + "\n".join(str(p) for p in existing)
        )


def tag_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}"


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def logit_np(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(prob / (1.0 - prob))


def format_float(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def format_percent(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def run_tta_predictions(args, device: str, coverage_tag: str):
    amp_enabled = args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    print(f"Loading ConvNeXt checkpoint: {args.ckpt}")
    model, ckpt, backbone, local_weights = load_model_from_checkpoint(
        args.ckpt, device)
    val_auc = ckpt.get("val_metrics", {}).get("auc")
    if val_auc is None:
        print(f"Backbone: {backbone} | best val AUC unavailable")
    else:
        print(f"Backbone: {backbone} | best val AUC {val_auc:.4f}")
    print(f"Local weights: {local_weights}")

    verify_no_leakage_mt()
    val_rows = load_mt_split("val")
    test_rows = load_mt_split("test")
    val_loader = DataLoader(
        MeniscusDatasetMT16SV2(val_rows, augment=False),
        batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        MeniscusDatasetMT16SV2(test_rows, augment=False),
        batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
        pin_memory=False,
    )
    tta_pipeline = build_tta_pipeline(no_tta=False)
    print(f"TTA transforms: {[name for name, _ in tta_pipeline]}")

    print("\nVALIDATION PREDICTIONS")
    val_prob, val_y, _val_sev_prob, _val_sev_y = predict_with_tta(
        model, val_loader, device, tta_pipeline,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )
    print("\nTEST PREDICTIONS")
    torch.cuda.empty_cache()
    test_prob, test_y, _test_sev_prob, _test_sev_y = predict_with_tta(
        model, test_loader, device, tta_pipeline,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )

    val_df = make_prediction_frame(val_rows, val_prob, val_y, args.threshold)
    test_df = make_prediction_frame(test_rows, test_prob, test_y, args.threshold)
    val_pred_csv = cfg.RESULTS / f"{coverage_tag}_val_predictions.csv"
    test_pred_csv = cfg.RESULTS / f"{coverage_tag}_test_predictions.csv"
    val_df.to_csv(val_pred_csv, index=False)
    test_df.to_csv(test_pred_csv, index=False)

    return {
        "model": model,
        "ckpt": ckpt,
        "backbone": str(backbone),
        "local_weights": str(local_weights),
        "tta_names": [name for name, _ in tta_pipeline],
        "val_rows": val_rows,
        "test_rows": test_rows,
        "val_df": val_df,
        "test_df": test_df,
        "val_pred_csv": val_pred_csv,
        "test_pred_csv": test_pred_csv,
    }


def write_coverage_markdown(path: Path, payload: dict):
    lines = [
        "# ConvNeXt Validation-Locked Coverage",
        "",
        "Model: `ConvNeXt base safe_strong 16-slice tri-plane multitask + 7x TTA`.",
        "",
        "Confidence: `max(p, 1-p)` at threshold `0.5`. Cutoffs were selected "
        "on validation only and applied unchanged to test.",
        "",
        "## Test Results",
        "",
        "| Target val coverage | Val cutoff | Test coverage | Test AUC | Test PR-AUC | Test F1 | Test sens/spec | Test error | Deferred n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["coverage_results"]:
        test = row["test"]
        cov = test["covered"]
        lines.append(
            f"| {row['target_coverage']:.0%} | {row['confidence_cutoff']:.4f} | "
            f"{test['coverage']:.1%} | {cov['auc']:.4f} | {cov['pr_auc']:.4f} | "
            f"{cov['f1']:.4f} | {cov['recall_sensitivity']:.3f}/{cov['specificity']:.3f} | "
            f"{cov['error_rate']:.3f} | {test['n_deferred']} |"
        )
    full = payload["full_coverage_test"]
    lines.extend([
        "",
        "## Full-Coverage Reference",
        "",
        f"- Test AUC: `{full['auc']:.4f}`",
        f"- Test PR-AUC: `{full['pr_auc']:.4f}`",
        f"- Test F1: `{full['f1']:.4f}`",
        f"- Test sensitivity/specificity: `{full['recall_sensitivity']:.3f}/{full['specificity']:.3f}`",
        f"- Test error rate: `{full['error_rate']:.3f}`",
        "",
        "## Artifacts",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def run_coverage(args, pred_info, coverage_tag: str):
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figures_dir = cfg.PROJECT / "figures"
    out_json = metrics_dir / f"{coverage_tag}.json"
    out_csv = metrics_dir / f"{coverage_tag}.csv"
    out_md = logs_dir / f"{coverage_tag}.md"
    out_fig = figures_dir / f"{coverage_tag}.png"

    val_df = pred_info["val_df"]
    test_df = pred_info["test_df"]
    full_val = metrics_for_subset(val_df, args.threshold)
    full_test = metrics_for_subset(test_df, args.threshold)
    coverage_rows = []
    flat_rows = []
    for target in args.coverages:
        cutoff = cutoff_for_target_coverage(val_df, target)
        val_result = coverage_result("val", val_df, cutoff, args.threshold)
        test_result = coverage_result("test", test_df, cutoff, args.threshold)
        row = {
            "target_coverage": float(target),
            "confidence_cutoff": float(cutoff),
            "val": val_result,
            "test": test_result,
        }
        coverage_rows.append(row)
        for split_name, result in (("val", val_result), ("test", test_result)):
            covered = result["covered"]
            flat_rows.append({
                "target_coverage": target,
                "split": split_name,
                "confidence_cutoff": cutoff,
                "actual_coverage": result["coverage"],
                "n_total": result["n_total"],
                "n_covered": result["n_covered"],
                "n_deferred": result["n_deferred"],
                "auc": covered["auc"],
                "pr_auc": covered["pr_auc"],
                "accuracy": covered["accuracy"],
                "error_rate": covered["error_rate"],
                "balanced_accuracy": covered["balanced_accuracy"],
                "f1": covered["f1"],
                "precision": covered["precision"],
                "sensitivity": covered["recall_sensitivity"],
                "specificity": covered["specificity"],
                "npv": covered["npv"],
                "tp": covered["confusion"]["tp"],
                "fp": covered["confusion"]["fp"],
                "fn": covered["confusion"]["fn"],
                "tn": covered["confusion"]["tn"],
            })

    payload = {
        "tag": coverage_tag,
        "model": str(args.ckpt),
        "backbone": pred_info["backbone"],
        "local_weights": pred_info["local_weights"],
        "threshold": args.threshold,
        "confidence_definition": "max(prob, 1-prob)",
        "coverage_targets": [float(value) for value in args.coverages],
        "tta_names": pred_info["tta_names"],
        "full_coverage_val": full_val,
        "full_coverage_test": full_test,
        "coverage_results": coverage_rows,
        "artifacts": {
            "json": str(out_json),
            "csv": str(out_csv),
            "markdown": str(out_md),
            "figure": str(out_fig),
            "val_predictions": str(pred_info["val_pred_csv"]),
            "test_predictions": str(pred_info["test_pred_csv"]),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    write_coverage_markdown(out_md, payload)
    plot_coverage(out_fig, coverage_rows, full_test)
    return payload


def write_calibration_markdown(path: Path, payload: dict):
    lines = [
        "# ConvNeXt Calibration",
        "",
        "Model: `ConvNeXt base safe_strong 16-slice tri-plane multitask + 7x TTA`.",
        "",
        f"Temperature was fit on validation only: `T = {payload['temperature']:.4f}`.",
        "",
        "## Metrics",
        "",
        "| Split | Variant | Brier | ECE | MCE | NLL | AUC | PR-AUC | F1 @0.5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("val", "test"):
        split_label = "Validation" if split_name == "val" else "Test"
        for variant_name in ("raw", "temperature_scaled"):
            metrics = payload["metrics"][split_name][variant_name]
            label = "Raw" if variant_name == "raw" else "Temperature scaled"
            lines.append(
                f"| {split_label} | {label} | "
                f"{metrics['brier']:.4f} | {metrics['ece']:.4f} | "
                f"{metrics['mce']:.4f} | {metrics['nll']:.4f} | "
                f"{metrics['auc']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{metrics['f1_at_0p5']:.4f} |"
            )
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def write_calibration_caption(path: Path, payload: dict):
    test_raw = payload["metrics"]["test"]["raw"]
    test_temp = payload["metrics"]["test"]["temperature_scaled"]
    text = (
        "Figure. Calibration of the ConvNeXt base 16-slice safe-strong "
        "meniscus tear classifier evaluated with 7x test-time augmentation. "
        "(A) Reliability curves on the held-out test set before and after "
        "validation-fitted temperature scaling. (B) Per-bin calibration gap. "
        "(C) Distribution of temperature-scaled tear probabilities by true "
        "class. (D) Test calibration metrics. Temperature was fitted using "
        f"validation data only (T={payload['temperature']:.3f}). On test, "
        f"temperature scaling changed Brier score from {test_raw['brier']:.3f} "
        f"to {test_temp['brier']:.3f}, ECE from {test_raw['ece']:.3f} to "
        f"{test_temp['ece']:.3f}, and NLL from {test_raw['nll']:.3f} to "
        f"{test_temp['nll']:.3f}."
    )
    path.write_text(text + "\n")


def run_calibration(args, pred_info, calibration_tag: str):
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figures_dir = cfg.PROJECT / "figures"
    out_json = metrics_dir / f"{calibration_tag}.json"
    metrics_csv = metrics_dir / f"{calibration_tag}_metrics.csv"
    bins_csv = metrics_dir / f"{calibration_tag}_reliability_bins.csv"
    calibrated_pred_csv = cfg.RESULTS / f"{calibration_tag}_test_predictions.csv"
    out_md = logs_dir / f"{calibration_tag}.md"
    caption_path = logs_dir / f"{calibration_tag}_figure_caption.txt"
    fig_path = figures_dir / f"{calibration_tag}.png"
    pdf_path = figures_dir / f"{calibration_tag}.pdf"

    val_df = pred_info["val_df"].copy()
    test_df = pred_info["test_df"].copy()
    temperature, val_temp_nll = fit_temperature(
        val_df["prob"].to_numpy(), val_df["label"].to_numpy())
    print(f"Fitted ConvNeXt validation temperature: {temperature:.4f}")
    print(f"Validation NLL after temperature scaling: {val_temp_nll:.4f}")

    for df in (val_df, test_df):
        df["prob_raw"] = df["prob"].astype(float).clip(1e-6, 1 - 1e-6)
        df["prob_temperature_scaled"] = apply_temperature(
            df["prob_raw"].to_numpy(), temperature)
        df["pred_temperature_scaled_0p5"] = (
            df["prob_temperature_scaled"] >= args.threshold).astype(int)

    metrics_rows = []
    bin_rows = []
    payload_metrics = {"val": {}, "test": {}}
    for split_name, df in (("val", val_df), ("test", test_df)):
        y = df["label"].to_numpy()
        for variant, column in (
            ("raw", "prob_raw"),
            ("temperature_scaled", "prob_temperature_scaled"),
        ):
            prob = df[column].to_numpy()
            metrics = calibration_metrics_for(y, prob, args.threshold, args.bins)
            payload_metrics[split_name][variant] = metrics
            metrics_rows.append({"split": split_name, "variant": variant, **metrics})
            bin_rows.extend(build_bin_table(
                split_name, variant, y, prob, args.bins))

    metrics_df = pd.DataFrame(metrics_rows)
    bins_df = pd.DataFrame(bin_rows)
    test_out = test_df[[
        "ID", "SIDE", "label", "prob_raw", "prob_temperature_scaled",
        "pred_0p5", "pred_temperature_scaled_0p5",
    ]].copy()
    test_out.to_csv(calibrated_pred_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    bins_df.to_csv(bins_csv, index=False)

    payload = {
        "tag": calibration_tag,
        "model": str(args.ckpt),
        "val_csv": str(pred_info["val_pred_csv"]),
        "test_csv": str(pred_info["test_pred_csv"]),
        "temperature": float(temperature),
        "temperature_fit": {
            "split": "validation",
            "objective": "binary_nll",
            "note": "Fit to logits reconstructed from TTA-averaged probabilities.",
        },
        "threshold": args.threshold,
        "n_bins": args.bins,
        "metrics": payload_metrics,
        "artifacts": {
            "json": str(out_json),
            "metrics_csv": str(metrics_csv),
            "bins_csv": str(bins_csv),
            "calibrated_test_predictions": str(calibrated_pred_csv),
            "markdown": str(out_md),
            "figure": str(fig_path),
            "figure_pdf": str(pdf_path),
            "caption": str(caption_path),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_calibration_markdown(out_md, payload)
    write_calibration_caption(caption_path, payload)
    plot_calibration_figure(
        fig_path, pdf_path, bins_df, metrics_df, test_df, args.bins)
    return payload, metrics_df, bins_df


def write_conformal_markdown(path: Path, payload: dict, metrics_df: pd.DataFrame):
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values(
        "target_coverage")
    lines = [
        "# ConvNeXt Conformal-Style Prediction Sets",
        "",
        f"- Tag: `{payload['tag']}`",
        f"- Temperature: `{payload['temperature']:.4f}`",
        "- Method: binary split-conformal style label sets using validation "
        "nonconformity scores `1 - P_temperature_scaled(true label)`.",
        "- Reporting note: the validation set was also used for checkpoint "
        "selection and temperature scaling, so this remains a validation-locked "
        "conformal-style/defer analysis.",
        "- Ambiguous `{no tear, tear}` and empty `{}` sets are both counted as "
        "`defer`.",
        "",
        "## Test Results",
        "",
        "| Target coverage | qhat | Empirical coverage | Singleton rate | Defer rate | Singleton accuracy | Singleton F1 | Sens | Spec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_df.to_dict("records"):
        lines.append(
            f"| {format_percent(row['target_coverage'])} | "
            f"{row['qhat']:.4f} | "
            f"{format_percent(row['empirical_coverage'])} | "
            f"{format_percent(row['singleton_rate'])} | "
            f"{format_percent(row['defer_rate'])} | "
            f"{format_percent(row['singleton_accuracy'])} | "
            f"{format_percent(row['singleton_f1'])} | "
            f"{format_percent(row['singleton_sensitivity'])} | "
            f"{format_percent(row['singleton_specificity'])} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def write_conformal_caption(path: Path, payload: dict, metrics_df: pd.DataFrame):
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values(
        "target_coverage")
    parts = []
    for row in test_df.to_dict("records"):
        parts.append(
            f"{format_percent(row['target_coverage'])} target: "
            f"{format_percent(row['empirical_coverage'])} empirical coverage, "
            f"{format_percent(row['singleton_rate'])} singleton outputs"
        )
    text = (
        "Figure. Validation-locked conformal-style uncertainty analysis for "
        "the ConvNeXt base 16-slice safe-strong model with 7x test-time "
        "augmentation and temperature-scaled probabilities. (A) Empirical test "
        "coverage compared with nominal coverage. (B) Prediction-set "
        "composition. (C) Performance among singleton predictions only. "
        "(D) Temperature-scaled probability distribution with the 90% target "
        "decision regions. " + " ".join(parts) + "."
    )
    path.write_text(text + "\n")


def run_conformal(args, pred_info, calibration_payload: dict, conformal_tag: str):
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figures_dir = cfg.PROJECT / "figures"
    out_json = metrics_dir / f"{conformal_tag}.json"
    metrics_csv = metrics_dir / f"{conformal_tag}_metrics.csv"
    test_predictions_csv = cfg.RESULTS / f"{conformal_tag}_test_predictions.csv"
    val_scores_csv = metrics_dir / f"{conformal_tag}_validation_scores.csv"
    out_md = logs_dir / f"{conformal_tag}.md"
    caption_path = logs_dir / f"{conformal_tag}_figure_caption.txt"
    fig_path = figures_dir / f"{conformal_tag}.png"
    pdf_path = figures_dir / f"{conformal_tag}.pdf"

    temperature = float(calibration_payload["temperature"])
    val_df = pred_info["val_df"].copy()
    test_df = pred_info["test_df"].copy()
    for df in (val_df, test_df):
        df["prob_raw"] = df["prob"].astype(float).clip(1e-6, 1 - 1e-6)
        df["prob_temperature_scaled"] = apply_temperature(
            df["prob_raw"].to_numpy(), temperature)

    val_scores = true_label_scores(val_df)
    val_score_df = val_df[[
        "ID", "SIDE", "label", "prob_raw", "prob_temperature_scaled"
    ]].copy()
    val_score_df["nonconformity_score"] = val_scores
    val_score_df.to_csv(val_scores_csv, index=False)

    alphas = sorted(set(float(a) for a in args.alphas))
    long_rows = []
    metric_rows = []
    qhats = {}
    for alpha in alphas:
        qhat = conformal_quantile(val_scores, alpha)
        qhats[str(alpha)] = qhat
        for split_name, df in (("val", val_df), ("test", test_df)):
            rows = build_prediction_rows(df, alpha, qhat)
            rows.insert(0, "split", split_name)
            metric_rows.append(summarize_split(rows, split_name))
            if split_name == "test":
                long_rows.append(rows)

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["split", "target_coverage"])
    long_test = pd.concat(long_rows, ignore_index=True)
    metrics_df.to_csv(metrics_csv, index=False)
    long_test.to_csv(test_predictions_csv, index=False)

    payload = {
        "tag": conformal_tag,
        "val_csv": str(pred_info["val_pred_csv"]),
        "test_csv": str(pred_info["test_pred_csv"]),
        "calibration_json": calibration_payload["artifacts"]["json"],
        "temperature": temperature,
        "alphas": alphas,
        "target_coverages": [1.0 - a for a in alphas],
        "quantiles": qhats,
        "method": {
            "nonconformity_score": "1 - P_temperature_scaled(true_label)",
            "prediction_rule": (
                "include candidate label y if 1 - P_temperature_scaled(y) <= qhat"
            ),
            "defer_definition": "ambiguous and empty prediction sets",
            "caveat": (
                "Validation split was also used for checkpoint selection and "
                "temperature scaling; report as validation-locked "
                "conformal-style analysis unless a dedicated calibration split "
                "is created."
            ),
        },
        "artifacts": {
            "json": str(out_json),
            "metrics_csv": str(metrics_csv),
            "test_predictions_csv": str(test_predictions_csv),
            "validation_scores_csv": str(val_scores_csv),
            "markdown": str(out_md),
            "caption": str(caption_path),
            "figure": str(fig_path),
            "figure_pdf": str(pdf_path),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_conformal_markdown(out_md, payload, metrics_df)
    write_conformal_caption(caption_path, payload, metrics_df)
    plot_conformal_figure(fig_path, pdf_path, metrics_df, long_test)
    return payload, metrics_df, long_test


def get_convnext_target_layer(net: nn.Module) -> nn.Module:
    stages = getattr(net, "stages", None)
    if stages is not None and len(stages) > 0:
        return stages[-1]
    for name in ("layer4", "blocks"):
        layer = getattr(net, name, None)
        if layer is not None:
            return layer
    last_conv = None
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not locate a convolutional Grad-CAM target layer")
    return last_conv


class ConvNeXtGradCAMCollector:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.handles = []
        for plane in cfg.SEQUENCES:
            layer = get_convnext_target_layer(model.branches[plane].net)
            self.handles.append(layer.register_forward_hook(
                self._make_forward_hook(plane)))

    def _make_forward_hook(self, plane: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.activations[plane] = output
            output.register_hook(
                lambda grad, plane=plane: self.gradients.__setitem__(
                    plane, grad))
        return hook

    def clear(self):
        self.activations.clear()
        self.gradients.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def move_x_to_device(x, device):
    return {key: value.unsqueeze(0).to(device) for key, value in x.items()}


def compute_case_gradcam(model, collector, sample, device: str,
                         target: str, target_pred: int):
    x, _y, _severity = sample
    x_dev = move_x_to_device(x, device)
    model.zero_grad(set_to_none=True)
    collector.clear()
    main_logits, _aux_logits = model(x_dev)
    logit = main_logits.squeeze()
    probability = float(torch.sigmoid(logit.detach()).cpu())
    pred = int(probability >= 0.5)
    score = logit if (target == "tear_logit" or int(target_pred) == 1) else -logit
    score.backward()

    cams = {}
    for plane in cfg.SEQUENCES:
        acts = collector.activations[plane]
        grads = collector.gradients[plane]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        cam = normalize_cam(cam)
        n_slices = x[plane].shape[0]
        cam = cam.reshape(n_slices, cam.shape[-2], cam.shape[-1])
        resized = []
        height, width = x[plane].shape[-2:]
        for idx in range(n_slices):
            resized.append(resize_cam_np(
                cam[idx].detach().cpu().numpy(), (width, height)))
        cams[plane] = np.stack(resized, axis=0)
    return cams, probability, pred


def write_gradcam_caption(path: Path, payload: dict):
    target = (
        "the predicted class" if payload["target"] == "predicted_class"
        else "the tear logit"
    )
    text = (
        "Figure. Grad-CAM visual plausibility examples for the ConvNeXt base "
        "16-slice safe-strong model. Panels A-D show representative true "
        "positive, true negative, false positive, and false negative test "
        "knees, respectively. For each knee, the slice with the strongest "
        "Grad-CAM response is shown separately for DESS, coronal MPR, and "
        f"axial MPR planes. Heatmaps explain {target}; warmer colors indicate "
        "stronger contribution to the selected score. This is a qualitative "
        "sanity check, not lesion segmentation."
    )
    path.write_text(text + "\n")


def write_gradcam_markdown(path: Path, payload: dict, summary_df: pd.DataFrame):
    lines = [
        "# ConvNeXt Grad-CAM Plausibility Review",
        "",
        f"- Tag: `{payload['tag']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Target score: `{payload['target']}`",
        f"- Review cases: `{payload['n_review_cases']}` "
        f"({payload['max_per_group']} per TP/TN/FP/FN group, when available)",
        "- Weak quantitative metrics are foreground/centrality sanity checks, "
        "not localization accuracy.",
        "",
        "## Summary By Outcome",
        "",
        "| Outcome | Cases | Mean foreground mass | Peak on foreground | Mean central-70 mass |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_df.to_dict("records"):
        lines.append(
            f"| {row['outcome_group']} | {int(row['n_cases'])} | "
            f"{row['mean_cam_mass_foreground']:.3f} | "
            f"{row['peak_on_foreground_rate']:.3f} | "
            f"{row['mean_cam_mass_central_70']:.3f} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def run_gradcam(args, pred_info, gradcam_tag: str, device: str):
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figures_dir = cfg.PROJECT / "figures"
    out_json = metrics_dir / f"{gradcam_tag}.json"
    selected_csv = metrics_dir / f"{gradcam_tag}_selected_cases.csv"
    example_csv = metrics_dir / f"{gradcam_tag}_figure_examples.csv"
    plane_metrics_csv = metrics_dir / f"{gradcam_tag}_plane_plausibility.csv"
    summary_csv = metrics_dir / f"{gradcam_tag}_summary.csv"
    out_md = logs_dir / f"{gradcam_tag}.md"
    caption_path = logs_dir / f"{gradcam_tag}_figure_caption.txt"
    fig_path = figures_dir / f"{gradcam_tag}.png"
    pdf_path = figures_dir / f"{gradcam_tag}.pdf"

    predictions = pred_info["test_df"][[
        "ID", "SIDE", "label", "prob", "pred_0p5", "confidence"
    ]].copy()
    test_rows = pred_info["test_rows"]
    merged = merge_test_rows_with_predictions(test_rows, predictions)
    review_rows = select_review_cases(merged, args.max_gradcam_per_group)
    review_rows.to_csv(selected_csv, index=False)

    ds = MeniscusDatasetMT16SV2(review_rows, augment=False)
    model = pred_info["model"]
    model.eval()
    collector = ConvNeXtGradCAMCollector(model)
    plane_metric_rows = []
    case_artifacts = {}

    try:
        for idx, row in review_rows.reset_index(drop=True).iterrows():
            sample = ds[idx]
            cams, prob_single_pass, pred_single_pass = compute_case_gradcam(
                model, collector, sample, device, args.gradcam_target,
                int(row["pred_0p5"]))
            case_key = f"{row['ID']}_{int(row['SIDE'])}"
            case_artifacts.setdefault(case_key, {"planes": {}})
            for plane in cfg.SEQUENCES:
                metrics = summarize_cam_plane(
                    case_key=case_key,
                    group=row["outcome_group"],
                    plane=plane,
                    stack=sample[0][plane],
                    cam_stack=cams[plane],
                    probability=float(row["prob"]),
                    label=int(row["label"]),
                    pred=int(row["pred_0p5"]),
                )
                metrics["prob_single_pass"] = prob_single_pass
                metrics["pred_single_pass"] = pred_single_pass
                plane_metric_rows.append(metrics)
                slice_idx = metrics["top_slice"]
                case_artifacts[case_key]["planes"][plane] = {
                    "slice": slice_idx,
                    "image": sample[0][plane][slice_idx, 0].detach().cpu().numpy(),
                    "cam": cams[plane][slice_idx],
                }
            torch.cuda.empty_cache()
    finally:
        collector.close()

    metrics_df = pd.DataFrame(plane_metric_rows)
    summary_df = summarize_plausibility(metrics_df)
    metrics_df.to_csv(plane_metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    examples = select_plausible_example_cases(review_rows, metrics_df)
    examples.to_csv(example_csv, index=False)
    render_case_figure(fig_path, pdf_path, examples, case_artifacts,
                       args.gradcam_target)

    payload = {
        "tag": gradcam_tag,
        "checkpoint": str(args.ckpt),
        "predictions_csv": str(pred_info["test_pred_csv"]),
        "target": args.gradcam_target,
        "max_per_group": args.max_gradcam_per_group,
        "n_review_cases": int(len(review_rows)),
        "n_example_cases": int(len(examples)),
        "example_selection": "highest weak plausibility score within each TP/TN/FP/FN group",
        "outcome_counts": review_rows["outcome_group"].value_counts().to_dict(),
        "artifacts": {
            "json": str(out_json),
            "selected_cases_csv": str(selected_csv),
            "figure_examples_csv": str(example_csv),
            "plane_plausibility_csv": str(plane_metrics_csv),
            "summary_csv": str(summary_csv),
            "markdown": str(out_md),
            "caption": str(caption_path),
            "figure": str(fig_path),
            "figure_pdf": str(pdf_path),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_gradcam_caption(caption_path, payload)
    write_gradcam_markdown(out_md, payload, summary_df)
    return payload, summary_df


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_resnet_reference(args):
    return {
        "tta": load_json(args.resnet_tta_json),
        "coverage": load_json(args.resnet_coverage_json),
        "calibration": load_json(args.resnet_calibration_json),
        "conformal": pd.read_csv(args.resnet_conformal_csv),
        "gradcam": pd.read_csv(args.resnet_gradcam_summary_csv),
    }


def build_comparison_tables(resnet_ref, coverage_payload, calibration_payload,
                            conformal_df, gradcam_summary_df):
    conv_full = coverage_payload["full_coverage_test"]
    resnet_full = resnet_ref["tta"]["test"]["binary_threshold_0.5"]
    overall_rows = [
        {
            "model": "ResNet50 safe_strong + 7x TTA",
            "auc": resnet_full["auc"],
            "pr_auc": resnet_full["pr_auc"],
            "f1": resnet_full["f1"],
            "sensitivity": resnet_full["sensitivity"],
            "specificity": resnet_full["specificity"],
            "severity_acc": resnet_ref["tta"]["test"]["severity"]["sev_acc"],
        },
        {
            "model": "ConvNeXt base safe_strong + 7x TTA",
            "auc": conv_full["auc"],
            "pr_auc": conv_full["pr_auc"],
            "f1": conv_full["f1"],
            "sensitivity": conv_full["recall_sensitivity"],
            "specificity": conv_full["specificity"],
            "severity_acc": np.nan,
        },
    ]

    coverage_rows = []
    for model_name, payload in (
        ("ResNet50 safe_strong + 7x TTA", resnet_ref["coverage"]),
        ("ConvNeXt base safe_strong + 7x TTA", coverage_payload),
    ):
        for row in payload["coverage_results"]:
            test = row["test"]
            cov = test["covered"]
            coverage_rows.append({
                "model": model_name,
                "target_val_coverage": row["target_coverage"],
                "confidence_cutoff": row["confidence_cutoff"],
                "actual_test_coverage": test["coverage"],
                "n_deferred": test["n_deferred"],
                "auc": cov["auc"],
                "pr_auc": cov["pr_auc"],
                "f1": cov["f1"],
                "sensitivity": cov["recall_sensitivity"],
                "specificity": cov["specificity"],
                "error_rate": cov["error_rate"],
            })

    calibration_rows = []
    for model_name, payload in (
        ("ResNet50 safe_strong + 7x TTA", resnet_ref["calibration"]),
        ("ConvNeXt base safe_strong + 7x TTA", calibration_payload),
    ):
        for variant in ("raw", "temperature_scaled"):
            metrics = payload["metrics"]["test"][variant]
            calibration_rows.append({
                "model": model_name,
                "variant": variant,
                "temperature": payload["temperature"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "mce": metrics["mce"],
                "nll": metrics["nll"],
                "auc": metrics["auc"],
                "pr_auc": metrics["pr_auc"],
                "f1_at_0p5": metrics["f1_at_0p5"],
            })

    conv_conf = conformal_df[conformal_df["split"] == "test"].copy()
    res_conf = resnet_ref["conformal"][resnet_ref["conformal"]["split"] == "test"].copy()
    conformal_rows = []
    for model_name, df in (
        ("ResNet50 safe_strong + 7x TTA", res_conf),
        ("ConvNeXt base safe_strong + 7x TTA", conv_conf),
    ):
        for row in df.to_dict("records"):
            conformal_rows.append({
                "model": model_name,
                "target_coverage": row["target_coverage"],
                "empirical_coverage": row["empirical_coverage"],
                "singleton_rate": row["singleton_rate"],
                "defer_rate": row["defer_rate"],
                "singleton_accuracy": row["singleton_accuracy"],
                "singleton_f1": row["singleton_f1"],
                "singleton_sensitivity": row["singleton_sensitivity"],
                "singleton_specificity": row["singleton_specificity"],
            })

    gradcam_rows = []
    for model_name, df in (
        ("ResNet50 safe_strong + 7x TTA", resnet_ref["gradcam"]),
        ("ConvNeXt base safe_strong + 7x TTA", gradcam_summary_df),
    ):
        for row in df.to_dict("records"):
            gradcam_rows.append({
                "model": model_name,
                "outcome_group": row["outcome_group"],
                "n_cases": row["n_cases"],
                "mean_cam_mass_foreground": row["mean_cam_mass_foreground"],
                "peak_on_foreground_rate": row["peak_on_foreground_rate"],
                "mean_cam_mass_central_70": row["mean_cam_mass_central_70"],
            })

    return {
        "overall": pd.DataFrame(overall_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "conformal": pd.DataFrame(conformal_rows),
        "gradcam": pd.DataFrame(gradcam_rows),
    }


def plot_comparison_figure(path: Path, pdf_path: Path, tables: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
    })
    colors = {
        "ResNet50 safe_strong + 7x TTA": "#4c78a8",
        "ConvNeXt base safe_strong + 7x TTA": "#f58518",
    }
    short_labels = {
        "ResNet50 safe_strong + 7x TTA": "ResNet50",
        "ConvNeXt base safe_strong + 7x TTA": "ConvNeXt",
    }
    fig = plt.figure(figsize=(8.4, 6.2), dpi=300)
    grid = GridSpec(2, 2, figure=fig, wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    overall = tables["overall"].set_index("model")
    metrics = ["auc", "pr_auc", "f1"]
    x = np.arange(len(metrics))
    width = 0.36
    for offset, model_name in ((-width / 2, overall.index[0]),
                               (width / 2, overall.index[1])):
        ax_a.bar(x + offset, [overall.loc[model_name, m] for m in metrics],
                 width=width, color=colors[model_name],
                 label=short_labels[model_name])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["AUC", "PR-AUC", "F1"])
    ax_a.set_ylim(0.72, 0.94)
    ax_a.set_ylabel("Full-coverage test metric")
    ax_a.set_title("Discrimination")
    ax_a.legend(frameon=False, loc="upper right", ncol=1)
    ax_a.grid(True, axis="y", linestyle=":", alpha=0.45)

    coverage = tables["coverage"].copy()
    for model_name, sub in coverage.groupby("model"):
        sub = sub.sort_values("target_val_coverage")
        ax_b.plot(sub["target_val_coverage"], sub["auc"], marker="o",
                  color=colors[model_name], label=short_labels[model_name])
    ax_b.set_xlabel("Validation target coverage")
    ax_b.set_ylabel("Covered-subset AUC")
    ax_b.set_ylim(0.86, 0.96)
    ax_b.set_title("Selective prediction")
    target_ticks = sorted(coverage["target_val_coverage"].unique())
    ax_b.set_xticks(target_ticks)
    ax_b.set_xticklabels([f"{v:.0%}" for v in target_ticks])
    ax_b.legend(frameon=False, loc="lower left")
    ax_b.grid(True, linestyle=":", alpha=0.45)

    calibration = tables["calibration"].copy()
    test_temp = calibration[calibration["variant"] == "temperature_scaled"]
    x2 = np.arange(2)
    metric_names = ["brier", "ece"]
    for offset, model_name in ((-width / 2, overall.index[0]),
                               (width / 2, overall.index[1])):
        row = test_temp[test_temp["model"] == model_name].iloc[0]
        ax_c.bar(x2 + offset, [row[m] for m in metric_names],
                 width=width, color=colors[model_name],
                 label=short_labels[model_name])
    ax_c.set_xticks(x2)
    ax_c.set_xticklabels(["Brier", "ECE"])
    ax_c.set_ylabel("Lower is better")
    ax_c.set_title("Temperature-scaled calibration")
    ax_c.legend(frameon=False, loc="upper right")
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.45)

    conformal = tables["conformal"].copy()
    for model_name, sub in conformal.groupby("model"):
        sub = sub.sort_values("target_coverage")
        ax_d.plot(sub["target_coverage"], sub["singleton_rate"],
                  marker="o", color=colors[model_name],
                  label=short_labels[model_name])
    ax_d.set_xlabel("Target coverage")
    ax_d.set_ylabel("Singleton rate")
    ax_d.set_ylim(0.35, 1.02)
    ax_d.set_xticks(sorted(conformal["target_coverage"].unique()))
    ax_d.set_xticklabels([f"{v:.0%}" for v in sorted(conformal["target_coverage"].unique())])
    ax_d.set_title("Conformal-style defer")
    ax_d.legend(frameon=False, loc="lower left")
    ax_d.grid(True, linestyle=":", alpha=0.45)

    for label, ax in zip(("A", "B", "C", "D"), (ax_a, ax_b, ax_c, ax_d)):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    fig.savefig(path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_comparison_markdown(path: Path, tables: dict, artifacts: dict):
    overall = tables["overall"].copy()
    conv = overall[overall["model"].str.startswith("ConvNeXt")].iloc[0]
    res = overall[overall["model"].str.startswith("ResNet")].iloc[0]
    lines = [
        "# ConvNeXt vs ResNet Confidence Stack Comparison",
        "",
        "Models compared:",
        "",
        "- ResNet50 safe_strong 16-slice + 7x TTA: previous locked confidence-analysis model.",
        "- ConvNeXt base safe_strong 16-slice + 7x TTA: current best completed internal model.",
        "",
        "## Full-Coverage Test Metrics",
        "",
        "| Model | AUC | PR-AUC | F1 | Sens | Spec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['f1']:.4f} | {row['sensitivity']:.3f} | "
            f"{row['specificity']:.3f} |"
        )
    lines.extend([
        "",
        "## Key Differences",
        "",
        f"- ConvNeXt improves full-coverage AUC by `{conv['auc'] - res['auc']:+.4f}`.",
        f"- ConvNeXt improves full-coverage PR-AUC by `{conv['pr_auc'] - res['pr_auc']:+.4f}`.",
        f"- ConvNeXt improves full-coverage F1 by `{conv['f1'] - res['f1']:+.4f}`.",
        f"- ConvNeXt shifts the operating point toward specificity: sensitivity changes `{conv['sensitivity'] - res['sensitivity']:+.3f}`, specificity changes `{conv['specificity'] - res['specificity']:+.3f}`.",
        "",
        "## Coverage Snapshot",
        "",
        "| Model | Target val coverage | Actual test coverage | AUC | F1 | Deferred n |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in tables["coverage"].sort_values(
            ["target_val_coverage", "model"], ascending=[False, True]).to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['target_val_coverage']:.0%} | "
            f"{row['actual_test_coverage']:.1%} | {row['auc']:.4f} | "
            f"{row['f1']:.4f} | {int(row['n_deferred'])} |"
        )
    lines.extend([
        "",
        "## Calibration Snapshot",
        "",
        "| Model | Variant | Temperature | Brier | ECE | NLL |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in tables["calibration"].to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['temperature']:.4f} | "
            f"{row['brier']:.4f} | {row['ece']:.4f} | {row['nll']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "ConvNeXt is stronger on discrimination and PR-AUC, and its default threshold produces much higher specificity. Confidence/defer analyses should therefore be reported with explicit operating points rather than as a single universal threshold.",
        "",
        "## Artifacts",
        "",
    ])
    for table_name, table_path in artifacts["tables"].items():
        lines.append(f"- `{table_name}_csv`: `{table_path}`")
    for key in ("figure", "figure_pdf", "markdown", "caption"):
        lines.append(f"- `{key}`: `{artifacts[key]}`")
    path.write_text("\n".join(lines) + "\n")


def write_comparison_caption(path: Path, tables: dict):
    overall = tables["overall"].copy()
    conv = overall[overall["model"].str.startswith("ConvNeXt")].iloc[0]
    res = overall[overall["model"].str.startswith("ResNet")].iloc[0]
    text = (
        "Figure. Comparison of the previous ResNet50 safe-strong 16-slice "
        "model and the ConvNeXt base safe-strong 16-slice model, both evaluated "
        "with 7x test-time augmentation. (A) Full-coverage test discrimination "
        "metrics. (B) Validation-locked selective-prediction performance. "
        "(C) Temperature-scaled calibration metrics. (D) Singleton output rate "
        "for conformal-style prediction sets. ConvNeXt improved full-coverage "
        f"AUC from {res['auc']:.3f} to {conv['auc']:.3f}, PR-AUC from "
        f"{res['pr_auc']:.3f} to {conv['pr_auc']:.3f}, and F1 from "
        f"{res['f1']:.3f} to {conv['f1']:.3f}."
    )
    path.write_text(text + "\n")


def run_comparison(args, coverage_payload, calibration_payload,
                   conformal_df, gradcam_summary_df, comparison_tag: str):
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figures_dir = cfg.PROJECT / "figures"
    resnet_ref = read_resnet_reference(args)
    if gradcam_summary_df is None:
        gradcam_summary_df = pd.DataFrame(columns=[
            "outcome_group", "n_cases", "mean_cam_mass_foreground",
            "peak_on_foreground_rate", "mean_cam_mass_central_70",
        ])
    tables = build_comparison_tables(
        resnet_ref, coverage_payload, calibration_payload,
        conformal_df, gradcam_summary_df)

    out_prefix = metrics_dir / comparison_tag
    overall_csv = metrics_dir / f"{comparison_tag}_overall.csv"
    coverage_csv = metrics_dir / f"{comparison_tag}_coverage.csv"
    calibration_csv = metrics_dir / f"{comparison_tag}_calibration.csv"
    conformal_csv = metrics_dir / f"{comparison_tag}_conformal.csv"
    gradcam_csv = metrics_dir / f"{comparison_tag}_gradcam.csv"
    out_json = metrics_dir / f"{comparison_tag}.json"
    out_md = logs_dir / f"{comparison_tag}.md"
    caption_path = logs_dir / f"{comparison_tag}_figure_caption.txt"
    fig_path = figures_dir / f"{comparison_tag}.png"
    pdf_path = figures_dir / f"{comparison_tag}.pdf"

    tables["overall"].to_csv(overall_csv, index=False)
    tables["coverage"].to_csv(coverage_csv, index=False)
    tables["calibration"].to_csv(calibration_csv, index=False)
    tables["conformal"].to_csv(conformal_csv, index=False)
    tables["gradcam"].to_csv(gradcam_csv, index=False)

    payload = {
        "tag": comparison_tag,
        "tables": {
            "overall": str(overall_csv),
            "coverage": str(coverage_csv),
            "calibration": str(calibration_csv),
            "conformal": str(conformal_csv),
            "gradcam": str(gradcam_csv),
        },
        "figure": str(fig_path),
        "figure_pdf": str(pdf_path),
        "markdown": str(out_md),
        "caption": str(caption_path),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    plot_comparison_figure(fig_path, pdf_path, tables)
    write_comparison_markdown(out_md, tables, payload)
    write_comparison_caption(caption_path, tables)
    return payload, tables


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    (cfg.RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.PROJECT / "figures").mkdir(parents=True, exist_ok=True)

    coverage_tag = tag_name(args.tag, "coverage")
    calibration_tag = tag_name(args.tag, "calibration")
    conformal_tag = tag_name(args.tag, "conformal")
    gradcam_tag = tag_name(args.tag, "gradcam")
    comparison_tag = tag_name(args.tag, "vs_resnet_confidence_comparison")

    planned_outputs = [
        cfg.RESULTS / f"{coverage_tag}_val_predictions.csv",
        cfg.RESULTS / f"{coverage_tag}_test_predictions.csv",
        cfg.RESULTS / "metrics" / f"{coverage_tag}.json",
        cfg.RESULTS / "metrics" / f"{coverage_tag}.csv",
        cfg.RESULTS / "logs" / f"{coverage_tag}.md",
        cfg.PROJECT / "figures" / f"{coverage_tag}.png",
        cfg.RESULTS / "metrics" / f"{calibration_tag}.json",
        cfg.RESULTS / "metrics" / f"{calibration_tag}_metrics.csv",
        cfg.RESULTS / "metrics" / f"{calibration_tag}_reliability_bins.csv",
        cfg.RESULTS / "logs" / f"{calibration_tag}.md",
        cfg.PROJECT / "figures" / f"{calibration_tag}.png",
        cfg.PROJECT / "figures" / f"{calibration_tag}.pdf",
        cfg.RESULTS / "metrics" / f"{conformal_tag}.json",
        cfg.RESULTS / "metrics" / f"{conformal_tag}_metrics.csv",
        cfg.RESULTS / "logs" / f"{conformal_tag}.md",
        cfg.PROJECT / "figures" / f"{conformal_tag}.png",
        cfg.PROJECT / "figures" / f"{conformal_tag}.pdf",
        cfg.RESULTS / "metrics" / f"{comparison_tag}.json",
        cfg.RESULTS / "logs" / f"{comparison_tag}.md",
        cfg.PROJECT / "figures" / f"{comparison_tag}.png",
        cfg.PROJECT / "figures" / f"{comparison_tag}.pdf",
    ]
    if not args.skip_gradcam:
        planned_outputs.extend([
            cfg.RESULTS / "metrics" / f"{gradcam_tag}.json",
            cfg.RESULTS / "metrics" / f"{gradcam_tag}_summary.csv",
            cfg.RESULTS / "logs" / f"{gradcam_tag}.md",
            cfg.PROJECT / "figures" / f"{gradcam_tag}.png",
            cfg.PROJECT / "figures" / f"{gradcam_tag}.pdf",
        ])
    guard_outputs(planned_outputs, args.overwrite)

    pred_info = run_tta_predictions(args, device, coverage_tag)
    coverage_payload = run_coverage(args, pred_info, coverage_tag)
    calibration_payload, _calibration_df, _bins_df = run_calibration(
        args, pred_info, calibration_tag)
    conformal_payload, conformal_df, _long_test = run_conformal(
        args, pred_info, calibration_payload, conformal_tag)

    gradcam_payload = None
    gradcam_summary_df = None
    if not args.skip_gradcam:
        gradcam_payload, gradcam_summary_df = run_gradcam(
            args, pred_info, gradcam_tag, device)

    comparison_payload, tables = run_comparison(
        args, coverage_payload, calibration_payload, conformal_df,
        gradcam_summary_df, comparison_tag)

    print("\nCONVNEXT STACK SUMMARY")
    full = coverage_payload["full_coverage_test"]
    print(
        f"Full coverage: AUC {full['auc']:.4f} | PR-AUC {full['pr_auc']:.4f} | "
        f"F1 {full['f1']:.4f} | sens/spec "
        f"{full['recall_sensitivity']:.3f}/{full['specificity']:.3f}"
    )
    test_raw = calibration_payload["metrics"]["test"]["raw"]
    test_temp = calibration_payload["metrics"]["test"]["temperature_scaled"]
    print(
        f"Calibration: T={calibration_payload['temperature']:.4f} | "
        f"Brier {test_raw['brier']:.4f}->{test_temp['brier']:.4f} | "
        f"ECE {test_raw['ece']:.4f}->{test_temp['ece']:.4f}"
    )
    conf_test = conformal_df[conformal_df["split"] == "test"].sort_values(
        "target_coverage")
    for row in conf_test.to_dict("records"):
        print(
            f"Conformal target {row['target_coverage']:.0%}: "
            f"emp_cov {row['empirical_coverage']:.1%}, "
            f"singleton {row['singleton_rate']:.1%}, "
            f"defer {row['defer_rate']:.1%}"
        )
    print(f"Comparison report -> {comparison_payload['markdown']}")
    print(f"Comparison figure -> {comparison_payload['figure']}")


if __name__ == "__main__":
    main()
