"""Grad-CAM review for selected test knees."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import config as cfg
from dataset_multitask_16s_v2_aug import (
    MeniscusDatasetMT16SV2,
    load_mt_split,
    verify_no_leakage_mt,
)
from eval_convnext_confidence_stack import (
    ConvNeXtGradCAMCollector,
    compute_case_gradcam,
)
from eval_multitask_16s_v2_timm_exp_tta import load_model_from_checkpoint
from gradcam_utils import (
    central_mask,
    foreground_mask,
    masked_mass,
    merge_test_rows_with_predictions,
    render_case_figure,
    select_review_cases,
    summarize_plausibility,
)


DEFAULT_TAG = "convnext_safe_strong_16s_tta_20260629_gradcam_curated_v3"
DEFAULT_CKPT = (
    cfg.WEIGHTS /
    "exp_multitask_16s_v2_convnext_base_safe_strong_bs4_ckpt_20260628_best.pt"
)
DEFAULT_PREDICTIONS = (
    cfg.RESULTS /
    "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curated central-slice ConvNeXt Grad-CAM figure")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--predictions_csv", type=Path,
                        default=DEFAULT_PREDICTIONS)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--max_per_group", type=int, default=8)
    parser.add_argument("--central_start", type=int, default=4)
    parser.add_argument("--central_end", type=int, default=11)
    parser.add_argument("--target", choices=["predicted_class", "tear_logit"],
                        default="predicted_class")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def guard_outputs(paths, overwrite: bool):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing curated Grad-CAM outputs. Use a "
            "new --tag or pass --overwrite.\n" +
            "\n".join(str(path) for path in existing)
        )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "SIDE", "label", "prob", "pred_0p5"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["ID"] = df["ID"].astype(str)
    df["SIDE"] = df["SIDE"].astype(int)
    df["label"] = df["label"].astype(int)
    df["pred_0p5"] = df["pred_0p5"].astype(int)
    df["prob"] = df["prob"].astype(float)
    df["confidence"] = np.maximum(df["prob"], 1.0 - df["prob"])
    return df


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if not finite.any():
        return out
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    if hi - lo < 1e-8:
        out[finite] = 1.0
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def summarize_cam_plane_curated(case_key: str, group: str, plane: str,
                                stack: torch.Tensor, cam_stack: np.ndarray,
                                probability: float, label: int, pred: int,
                                central_start: int,
                                central_end: int) -> dict:
    n_slices = int(cam_stack.shape[0])
    strengths = cam_stack.reshape(n_slices, -1).mean(axis=1)
    global_top_idx = int(np.argmax(strengths))
    start = max(0, int(central_start))
    end = min(n_slices - 1, int(central_end))
    if start > end:
        start, end = 0, n_slices - 1
    candidates = np.arange(start, end + 1, dtype=int)

    rows = []
    for idx in candidates:
        image = stack[idx, 0].detach().cpu().numpy()
        cam = cam_stack[idx]
        fg = foreground_mask(image)
        center = central_mask(image.shape)
        peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        rows.append({
            "slice": int(idx),
            "cam_strength": float(strengths[idx]),
            "cam_mass_foreground": masked_mass(cam, fg),
            "cam_mass_central_70": masked_mass(cam, center),
            "peak_on_foreground": int(bool(fg[peak_y, peak_x])),
            "foreground_fraction": float(fg.mean()),
            "image_std": float(np.std(image)),
            "peak_y": int(peak_y),
            "peak_x": int(peak_x),
        })

    cand = pd.DataFrame(rows)
    cand["score"] = (
        0.40 * _normalize(cand["cam_strength"].to_numpy()) +
        0.20 * cand["cam_mass_foreground"].fillna(0.0).to_numpy() +
        0.20 * cand["cam_mass_central_70"].fillna(0.0).to_numpy() +
        0.10 * cand["peak_on_foreground"].to_numpy(dtype=float) +
        0.10 * _normalize(cand["image_std"].to_numpy())
    )
    best = cand.sort_values(["score", "cam_strength"], ascending=False).iloc[0]
    top_idx = int(best["slice"])

    return {
        "case_key": case_key,
        "outcome_group": group,
        "plane": plane,
        "label": int(label),
        "pred_0p5": int(pred),
        "prob": float(probability),
        "top_slice": top_idx,
        "global_top_slice": global_top_idx,
        "slice_changed_from_global": int(top_idx != global_top_idx),
        "central_start": int(start),
        "central_end": int(end),
        "top_slice_strength": float(best["cam_strength"]),
        "global_top_slice_strength": float(strengths[global_top_idx]),
        "curated_slice_score": float(best["score"]),
        "cam_mass_foreground": float(best["cam_mass_foreground"]),
        "cam_mass_central_70": float(best["cam_mass_central_70"]),
        "peak_on_foreground": int(best["peak_on_foreground"]),
        "foreground_fraction": float(best["foreground_fraction"]),
        "image_std": float(best["image_std"]),
        "peak_y": int(best["peak_y"]),
        "peak_x": int(best["peak_x"]),
    }


def select_curated_example_cases(review_rows: pd.DataFrame,
                                 metrics_df: pd.DataFrame) -> pd.DataFrame:
    case_scores = metrics_df.groupby("case_key", as_index=False).agg(
        mean_foreground=("cam_mass_foreground", "mean"),
        mean_central=("cam_mass_central_70", "mean"),
        peak_rate=("peak_on_foreground", "mean"),
        mean_image_std=("image_std", "mean"),
        mean_score=("curated_slice_score", "mean"),
        changed_planes=("slice_changed_from_global", "sum"),
    )
    case_scores["visual_score"] = (
        case_scores["mean_score"] +
        0.25 * case_scores["mean_foreground"] +
        0.25 * case_scores["mean_central"] +
        0.15 * case_scores["peak_rate"] +
        0.10 * _normalize(case_scores["mean_image_std"].to_numpy())
    )
    rows = review_rows.copy()
    rows["case_key"] = (
        rows["ID"].astype(str) + "_" + rows["SIDE"].astype(int).astype(str)
    )
    rows = rows.merge(case_scores, on="case_key", how="left")
    selected = []
    for group in ["TP", "TN", "FP", "FN"]:
        sub = rows[rows["outcome_group"] == group].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["visual_score", "confidence"], ascending=False)
        selected.append(sub.head(1))
    if not selected:
        raise ValueError("No TP/TN/FP/FN examples available for Grad-CAM figure")
    return pd.concat(selected, ignore_index=True)


def write_caption(path: Path, payload: dict):
    target = (
        "the predicted class" if payload["target"] == "predicted_class"
        else "the tear logit"
    )
    text = (
        "Figure. Curated Grad-CAM visual plausibility examples for the "
        "ConvNeXt base 16-slice safe-strong model with 7x test-time "
        "augmentation. Panels A-D show representative true positive, true "
        "negative, false positive, and false negative test knees. For each "
        "plane, the displayed slice was selected from the central slice window "
        f"{payload['central_start']}-{payload['central_end']} by combining "
        "Grad-CAM response, foreground mass, central-field mass, and image "
        f"content. Heatmaps explain {target}; warmer colors indicate stronger "
        "contribution to the selected score. This is a qualitative sanity "
        "check, not lesion segmentation or radiologist-verified localization."
    )
    path.write_text(text + "\n")


def write_markdown(path: Path, payload: dict, summary_df: pd.DataFrame):
    lines = [
        "# Curated ConvNeXt Grad-CAM Plausibility Review",
        "",
        f"- Tag: `{payload['tag']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Predictions: `{payload['predictions_csv']}`",
        f"- Target score: `{payload['target']}`",
        f"- Review cases: `{payload['n_review_cases']}` "
        f"({payload['max_per_group']} per TP/TN/FP/FN group, when available)",
        f"- Display-slice candidate window: `{payload['central_start']}` to "
        f"`{payload['central_end']}` inclusive.",
        "- Slice selection combines CAM strength, foreground CAM mass, "
        "central-field CAM mass, peak-on-foreground, and image-content "
        "contrast to avoid peripheral/empty slices.",
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
    lines.extend([
        "",
        f"Planes where the curated slice differed from the global strongest "
        f"CAM slice: `{payload['n_planes_changed_from_global']}` / "
        f"`{payload['n_plane_maps']}`.",
        "",
        "## Artifacts",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for Grad-CAM.")
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figures_dir = cfg.PROJECT / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / f"{args.tag}.json"
    selected_csv = metrics_dir / f"{args.tag}_selected_cases.csv"
    example_csv = metrics_dir / f"{args.tag}_figure_examples.csv"
    plane_metrics_csv = metrics_dir / f"{args.tag}_plane_plausibility.csv"
    summary_csv = metrics_dir / f"{args.tag}_summary.csv"
    out_md = logs_dir / f"{args.tag}.md"
    caption_path = logs_dir / f"{args.tag}_figure_caption.txt"
    figure_path = figures_dir / f"{args.tag}.png"
    figure_pdf_path = figures_dir / f"{args.tag}.pdf"
    guard_outputs(
        [out_json, selected_csv, example_csv, plane_metrics_csv, summary_csv,
         out_md, caption_path, figure_path, figure_pdf_path],
        args.overwrite,
    )

    verify_no_leakage_mt()
    predictions = load_predictions(args.predictions_csv)
    test_rows = load_mt_split("test")
    merged = merge_test_rows_with_predictions(test_rows, predictions)
    review_rows = select_review_cases(merged, args.max_per_group)
    review_rows.to_csv(selected_csv, index=False)

    print(f"Loading ConvNeXt checkpoint: {args.ckpt}")
    model, ckpt, backbone, local_weights = load_model_from_checkpoint(
        args.ckpt, device)
    val_auc = ckpt.get("val_metrics", {}).get("auc")
    if val_auc is not None:
        print(f"Backbone: {backbone} | best val AUC {val_auc:.4f}")
    print(f"Local weights: {local_weights}")

    ds = MeniscusDatasetMT16SV2(review_rows, augment=False)
    model.eval()
    collector = ConvNeXtGradCAMCollector(model)
    plane_metric_rows = []
    case_artifacts = {}

    try:
        for idx, row in review_rows.reset_index(drop=True).iterrows():
            sample = ds[idx]
            cams, prob_single_pass, pred_single_pass = compute_case_gradcam(
                model, collector, sample, device, args.target,
                int(row["pred_0p5"]))
            case_key = f"{row['ID']}_{int(row['SIDE'])}"
            case_artifacts.setdefault(case_key, {"planes": {}})
            for plane in cfg.SEQUENCES:
                metrics = summarize_cam_plane_curated(
                    case_key=case_key,
                    group=row["outcome_group"],
                    plane=plane,
                    stack=sample[0][plane],
                    cam_stack=cams[plane],
                    probability=float(row["prob"]),
                    label=int(row["label"]),
                    pred=int(row["pred_0p5"]),
                    central_start=args.central_start,
                    central_end=args.central_end,
                )
                metrics["prob_single_pass"] = prob_single_pass
                metrics["pred_single_pass"] = pred_single_pass
                plane_metric_rows.append(metrics)
                slice_idx = int(metrics["top_slice"])
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
    examples = select_curated_example_cases(review_rows, metrics_df)
    metrics_df.to_csv(plane_metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    examples.to_csv(example_csv, index=False)
    render_case_figure(figure_path, figure_pdf_path, examples,
                       case_artifacts, args.target)

    payload = {
        "tag": args.tag,
        "checkpoint": str(args.ckpt),
        "predictions_csv": str(args.predictions_csv),
        "backbone": str(backbone),
        "local_weights": str(local_weights),
        "target": args.target,
        "max_per_group": int(args.max_per_group),
        "central_start": int(args.central_start),
        "central_end": int(args.central_end),
        "n_review_cases": int(len(review_rows)),
        "n_example_cases": int(len(examples)),
        "n_plane_maps": int(len(metrics_df)),
        "n_planes_changed_from_global": int(
            metrics_df["slice_changed_from_global"].sum()),
        "example_selection": (
            "highest visual score within each TP/TN/FP/FN group after "
            "central-window slice selection"
        ),
        "outcome_counts": review_rows["outcome_group"].value_counts().to_dict(),
        "artifacts": {
            "json": str(out_json),
            "selected_cases_csv": str(selected_csv),
            "figure_examples_csv": str(example_csv),
            "plane_plausibility_csv": str(plane_metrics_csv),
            "summary_csv": str(summary_csv),
            "markdown": str(out_md),
            "caption": str(caption_path),
            "figure": str(figure_path),
            "figure_pdf": str(figure_pdf_path),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_caption(caption_path, payload)
    write_markdown(out_md, payload, summary_df)

    print("\nCURATED CONVNEXT GRAD-CAM SUMMARY")
    print(summary_df.to_string(index=False))
    print(
        "Changed displayed slice vs global strongest CAM in "
        f"{payload['n_planes_changed_from_global']}/{payload['n_plane_maps']} "
        "plane maps."
    )
    print(f"Saved curated figure -> {figure_path}")
    print(f"Saved curated caption -> {caption_path}")
    print(f"Saved curated report -> {out_md}")


if __name__ == "__main__":
    main()
