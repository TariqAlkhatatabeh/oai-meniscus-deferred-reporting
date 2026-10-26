"""Grad-CAM review helpers for the primary model."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config as cfg

def merge_test_rows_with_predictions(test_rows: pd.DataFrame,
                                     predictions: pd.DataFrame) -> pd.DataFrame:
    rows = test_rows.copy()
    rows["ID"] = rows["ID"].astype(str)
    rows["SIDE"] = rows["SIDE"].astype(int)
    merged = rows.merge(
        predictions[["ID", "SIDE", "prob", "pred_0p5", "confidence"]],
        on=["ID", "SIDE"], how="inner", suffixes=("", "_pred"))
    if len(merged) != len(predictions):
        print(
            f"  [warn] merged {len(merged)} rows but prediction CSV has "
            f"{len(predictions)} rows"
        )
    merged["outcome_group"] = np.select(
        [
            (merged["label"].astype(int) == 1) & (merged["pred_0p5"] == 1),
            (merged["label"].astype(int) == 0) & (merged["pred_0p5"] == 0),
            (merged["label"].astype(int) == 0) & (merged["pred_0p5"] == 1),
            (merged["label"].astype(int) == 1) & (merged["pred_0p5"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="other",
    )
    return merged

def select_review_cases(rows: pd.DataFrame, max_per_group: int) -> pd.DataFrame:
    selected = []
    for group in ["TP", "TN", "FP", "FN"]:
        sub = rows[rows["outcome_group"] == group].copy()
        sub = sub.sort_values("confidence", ascending=False)
        selected.append(sub.head(max_per_group))
    out = pd.concat(selected, ignore_index=True)
    return out.reset_index(drop=True)

def select_plausible_example_cases(review_rows: pd.DataFrame,
                                   metrics_df: pd.DataFrame) -> pd.DataFrame:
    case_scores = metrics_df.groupby("case_key", as_index=False).agg(
        mean_foreground=("cam_mass_foreground", "mean"),
        mean_central=("cam_mass_central_70", "mean"),
        peak_rate=("peak_on_foreground", "mean"),
    )
    case_scores["plausibility_score"] = (
        case_scores["mean_foreground"] +
        case_scores["mean_central"] +
        case_scores["peak_rate"]
    )
    rows = review_rows.copy()
    rows["case_key"] = rows["ID"].astype(str) + "_" + rows["SIDE"].astype(int).astype(str)
    rows = rows.merge(case_scores, on="case_key", how="left")
    selected = []
    for group in ["TP", "TN", "FP", "FN"]:
        sub = rows[rows["outcome_group"] == group].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["plausibility_score", "confidence"], ascending=False)
        selected.append(sub.head(1))
    if not selected:
        raise ValueError("No TP/TN/FP/FN examples available for Grad-CAM figure")
    return pd.concat(selected, ignore_index=True)

def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
    denom = cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return cam / denom

def resize_cam_np(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.uint8(np.clip(cam, 0.0, 1.0) * 255))
    img = img.resize(size, resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def foreground_mask(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    threshold = max(0.04, float(np.percentile(image, 10)))
    return image > threshold

def central_mask(shape: tuple[int, int], frac: float = 0.70) -> np.ndarray:
    h, w = shape
    margin_y = int(round((1.0 - frac) * h / 2.0))
    margin_x = int(round((1.0 - frac) * w / 2.0))
    mask = np.zeros((h, w), dtype=bool)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = True
    return mask

def masked_mass(cam: np.ndarray, mask: np.ndarray) -> float:
    total = float(cam.sum())
    if total <= 1e-8:
        return float("nan")
    return float(cam[mask].sum() / total)

def summarize_cam_plane(case_key: str, group: str, plane: str,
                        stack: torch.Tensor, cam_stack: np.ndarray,
                        probability: float, label: int, pred: int) -> dict:
    strengths = cam_stack.reshape(cam_stack.shape[0], -1).mean(axis=1)
    top_idx = int(np.argmax(strengths))
    image = stack[top_idx, 0].detach().cpu().numpy()
    cam = cam_stack[top_idx]
    fg = foreground_mask(image)
    center = central_mask(image.shape)
    peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {
        "case_key": case_key,
        "outcome_group": group,
        "plane": plane,
        "label": int(label),
        "pred_0p5": int(pred),
        "prob": float(probability),
        "top_slice": top_idx,
        "top_slice_strength": float(strengths[top_idx]),
        "cam_mass_foreground": masked_mass(cam, fg),
        "cam_mass_central_70": masked_mass(cam, center),
        "peak_on_foreground": int(bool(fg[peak_y, peak_x])),
        "peak_y": int(peak_y),
        "peak_x": int(peak_x),
    }

def render_case_figure(path: Path, pdf_path: Path, examples: pd.DataFrame,
                       case_artifacts: dict, target: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "figure.dpi": 300,
    })

    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, len(cfg.SEQUENCES),
                             figsize=(7.4, 2.05 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.16, right=0.99, top=0.94, bottom=0.04,
                        wspace=0.04, hspace=0.24)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    group_seq: dict[str, int] = {}
    plane_labels = {
        "dess": "DESS",
        "cor_mpr": "Coronal MPR",
        "ax_mpr": "Axial MPR",
    }

    for row_idx, row in examples.reset_index(drop=True).iterrows():
        case_key = f"{row['ID']}_{int(row['SIDE'])}"
        artifact = case_artifacts[case_key]
        for col_idx, plane in enumerate(cfg.SEQUENCES):
            ax = axes[row_idx, col_idx]
            plane_info = artifact["planes"][plane]
            image = plane_info["image"]
            cam = plane_info["cam"]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam, cmap="magma", vmin=0, vmax=1,
                      alpha=np.clip(cam * 0.65, 0.0, 0.65))
            ax.set_axis_off()
            ax.set_title(
                f"{plane_labels.get(plane, plane)} | slice {plane_info['slice']:02d}",
                pad=2,
            )
        label = int(row["label"])
        pred = int(row["pred_0p5"])
        prob = float(row["prob"])
        # A study-local case label replaces the OAI subject id. The research id is not a direct
        # identifier but it is linkable, and the release repository pseudonymises exactly these ids.
        group = str(row["outcome_group"])
        group_seq[group] = group_seq.get(group, 0) + 1
        text = (
            f"{letters[row_idx]}\n{group}-{group_seq[group]:02d}\n"
            f"SIDE {int(row['SIDE'])}\n"
            f"y={label}, pred={pred}\np={prob:.3f}"
        )
        axes[row_idx, 0].text(
            -0.54, 0.98, text, transform=axes[row_idx, 0].transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold"
        )

    title_target = "predicted class" if target == "predicted_class" else "tear logit"
    fig.suptitle(f"Grad-CAM examples for {title_target}", y=0.992,
                 fontsize=10, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def summarize_plausibility(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby("outcome_group", sort=False)
    rows = []
    for group, sub in grouped:
        rows.append({
            "outcome_group": group,
            "n_cases": int(sub["case_key"].nunique()),
            "n_plane_maps": int(len(sub)),
            "mean_cam_mass_foreground": float(sub["cam_mass_foreground"].mean()),
            "peak_on_foreground_rate": float(sub["peak_on_foreground"].mean()),
            "mean_cam_mass_central_70": float(sub["cam_mass_central_70"].mean()),
            "mean_top_slice_strength": float(sub["top_slice_strength"].mean()),
        })
    return pd.DataFrame(rows)
