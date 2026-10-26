"""Participant-clustered bootstrap intervals for test metrics."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import config as cfg


DEFAULT_CONVNEXT = (
    cfg.RESULTS /
    "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"
)
DEFAULT_RESNET = (
    cfg.RESULTS /
    "primary_model_safe_strong_16s_tta_coverage_20260628_test_predictions.csv"
)
DEFAULT_TAG = "convnext_safe_strong_16s_tta_20260629_bootstrap_ci"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CIs for test-set metrics")
    parser.add_argument("--convnext_csv", type=Path, default=DEFAULT_CONVNEXT)
    parser.add_argument("--resnet_csv", type=Path, default=DEFAULT_RESNET)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def guard_outputs(paths, overwrite: bool):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite bootstrap outputs. Use a new --tag or "
            "--overwrite.\n" + "\n".join(str(p) for p in existing)
        )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "SIDE", "label", "prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["ID"] = df["ID"].astype(str)
    df["SIDE"] = df["SIDE"].astype(int)
    df["label"] = df["label"].astype(int)
    df["prob"] = df["prob"].astype(float).clip(1e-6, 1 - 1e-6)
    return df.reset_index(drop=True)


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    y = df["label"].astype(int).to_numpy()
    p = df["prob"].astype(float).to_numpy()
    pred = (p >= threshold).astype(int)
    out = {}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
        out["balanced_accuracy"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    })
    return out


def sample_cluster_indices(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    ids = df["ID"].drop_duplicates().to_numpy()
    sampled = rng.choice(ids, size=len(ids), replace=True)
    index_by_id = {pid: np.where(df["ID"].to_numpy() == pid)[0] for pid in ids}
    return np.concatenate([index_by_id[pid] for pid in sampled])


def bootstrap_model(df: pd.DataFrame, threshold: float, n_boot: int,
                    rng: np.random.Generator) -> tuple[dict, dict[str, np.ndarray]]:
    point = compute_metrics(df, threshold)
    values = {key: [] for key in point if key not in {"tp", "fp", "fn", "tn"}}
    for _ in range(n_boot):
        idx = sample_cluster_indices(df, rng)
        metrics = compute_metrics(df.iloc[idx].reset_index(drop=True), threshold)
        for key in values:
            values[key].append(metrics[key])
    arrays = {key: np.asarray(value, dtype=float) for key, value in values.items()}
    return point, arrays


def align_pair(convnext: pd.DataFrame, resnet: pd.DataFrame) -> pd.DataFrame:
    left = convnext[["ID", "SIDE", "label", "prob"]].rename(
        columns={"prob": "prob_convnext"})
    right = resnet[["ID", "SIDE", "label", "prob"]].rename(
        columns={"prob": "prob_resnet", "label": "label_resnet"})
    merged = left.merge(right, on=["ID", "SIDE"], how="inner")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"Prediction rows do not align: convnext={len(left)} "
            f"resnet={len(right)} merged={len(merged)}"
        )
    if not (merged["label"].astype(int) == merged["label_resnet"].astype(int)).all():
        raise ValueError("Labels differ between paired prediction files")
    return merged.drop(columns=["label_resnet"]).reset_index(drop=True)


def metric_df_from_pair(pair: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    return pair[["ID", "SIDE", "label", prob_col]].rename(
        columns={prob_col: "prob"})


def bootstrap_difference(pair: pd.DataFrame, threshold: float, n_boot: int,
                         rng: np.random.Generator) -> tuple[dict, dict[str, np.ndarray]]:
    conv_df = metric_df_from_pair(pair, "prob_convnext")
    res_df = metric_df_from_pair(pair, "prob_resnet")
    conv_point = compute_metrics(conv_df, threshold)
    res_point = compute_metrics(res_df, threshold)
    metric_names = [key for key in conv_point if key not in {"tp", "fp", "fn", "tn"}]
    point = {
        key: float(conv_point[key] - res_point[key])
        for key in metric_names
    }
    ids = pair["ID"].drop_duplicates().to_numpy()
    index_by_id = {pid: np.where(pair["ID"].to_numpy() == pid)[0] for pid in ids}
    values = {key: [] for key in metric_names}
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([index_by_id[pid] for pid in sampled])
        sub = pair.iloc[idx].reset_index(drop=True)
        conv = compute_metrics(metric_df_from_pair(sub, "prob_convnext"), threshold)
        res = compute_metrics(metric_df_from_pair(sub, "prob_resnet"), threshold)
        for key in metric_names:
            values[key].append(conv[key] - res[key])
    arrays = {key: np.asarray(value, dtype=float) for key, value in values.items()}
    return point, arrays


def summarize_arrays(model_name: str, point: dict, arrays: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for metric, samples in arrays.items():
        finite = samples[np.isfinite(samples)]
        rows.append({
            "model": model_name,
            "metric": metric,
            "estimate": float(point[metric]),
            "ci_low": float(np.percentile(finite, 2.5)),
            "ci_high": float(np.percentile(finite, 97.5)),
            "bootstrap_se": float(np.std(finite, ddof=1)),
            "n_boot_valid": int(len(finite)),
        })
    return rows


def write_markdown(path: Path, metrics_df: pd.DataFrame,
                   diff_df: pd.DataFrame, payload: dict):
    selected = ["auc", "pr_auc", "f1", "sensitivity", "specificity", "brier"]
    lines = [
        "# Bootstrap Confidence Intervals",
        "",
        "- Bootstrap unit: participant ID; all knees from a sampled participant are kept together.",
        f"- Replicates: `{payload['n_boot']}`",
        f"- Threshold: `{payload['threshold']}`",
        "",
        "## Model Metrics",
        "",
        "| Model | Metric | Estimate | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in metrics_df[metrics_df["metric"].isin(selected)].to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['estimate']:.4f} | "
            f"{row['ci_low']:.4f}-{row['ci_high']:.4f} |"
        )
    lines.extend([
        "",
        "## Paired Difference: ConvNeXt - ResNet",
        "",
        "| Metric | Difference | 95% CI |",
        "|---|---:|---:|",
    ])
    for row in diff_df[diff_df["metric"].isin(selected)].to_dict("records"):
        lines.append(
            f"| {row['metric']} | {row['estimate']:+.4f} | "
            f"{row['ci_low']:+.4f}-{row['ci_high']:+.4f} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / f"{args.tag}.json"
    metrics_csv = metrics_dir / f"{args.tag}_model_metrics.csv"
    diff_csv = metrics_dir / f"{args.tag}_paired_differences.csv"
    out_md = logs_dir / f"{args.tag}.md"
    guard_outputs([out_json, metrics_csv, diff_csv, out_md], args.overwrite)

    convnext = load_predictions(args.convnext_csv)
    resnet = load_predictions(args.resnet_csv)
    pair = align_pair(convnext, resnet)
    rng = np.random.default_rng(args.seed)

    conv_point, conv_arrays = bootstrap_model(
        convnext, args.threshold, args.n_boot, rng)
    res_point, res_arrays = bootstrap_model(
        resnet, args.threshold, args.n_boot, rng)
    diff_point, diff_arrays = bootstrap_difference(
        pair, args.threshold, args.n_boot, rng)

    metric_rows = []
    metric_rows.extend(summarize_arrays("ConvNeXt safe_strong + 7x TTA",
                                        conv_point, conv_arrays))
    metric_rows.extend(summarize_arrays("ResNet50 safe_strong + 7x TTA",
                                        res_point, res_arrays))
    diff_rows = summarize_arrays("ConvNeXt - ResNet", diff_point, diff_arrays)
    metrics_df = pd.DataFrame(metric_rows)
    diff_df = pd.DataFrame(diff_rows).drop(columns=["model"])
    metrics_df.to_csv(metrics_csv, index=False)
    diff_df.to_csv(diff_csv, index=False)

    payload = {
        "tag": args.tag,
        "convnext_csv": str(args.convnext_csv),
        "resnet_csv": str(args.resnet_csv),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "threshold": float(args.threshold),
        "bootstrap_unit": "participant ID",
        "n_test_knees": int(len(convnext)),
        "n_test_participants": int(convnext["ID"].nunique()),
        "artifacts": {
            "json": str(out_json),
            "model_metrics_csv": str(metrics_csv),
            "paired_differences_csv": str(diff_csv),
            "markdown": str(out_md),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(out_md, metrics_df, diff_df, payload)

    print(f"Saved bootstrap model metrics -> {metrics_csv}")
    print(f"Saved bootstrap paired differences -> {diff_csv}")
    print(f"Saved report -> {out_md}")


if __name__ == "__main__":
    main()
