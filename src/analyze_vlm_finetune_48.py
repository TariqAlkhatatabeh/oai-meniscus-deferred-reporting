"""Summarize the QLoRA VLM evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune_dir", type=Path,
                   default=ROOT / "results" / "vlm_finetune_48_v1")
    p.add_argument("--finetune_csv", type=Path, default=None,
                   help="Optional explicit test-score CSV; overrides <finetune_dir>/test_scores.csv")
    p.add_argument("--best_info", type=Path, default=None,
                   help="Optional explicit best_info.json; overrides <finetune_dir>/best_info.json")
    p.add_argument("--zero_csv", type=Path,
                   default=ROOT / "results" / "vlm_eval" / "qwen25vl7b_planes3_predictions.csv")
    p.add_argument("--supervised_csv", type=Path,
                   default=ROOT / "results" / "convnext_safe_strong_16s_tta_20260629_calibration_test_predictions.csv")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--tag", default="vlm_finetune_48_locked_test_v1")
    return p.parse_args()


def metric(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def main():
    args = parse_args()
    fine_path = args.finetune_csv or (args.finetune_dir / "test_scores.csv")
    info_path = args.best_info or (args.finetune_dir / "best_info.json")
    if not fine_path.exists() or not info_path.exists():
        raise FileNotFoundError("Run locked test scoring before this analysis")

    fine = pd.read_csv(fine_path, dtype={"ID": str, "SIDE": str})
    zero = pd.read_csv(args.zero_csv, dtype={"ID": str, "SIDE": str})
    zero = zero.loc[zero["variant"].eq("planes3_montage"),
                    ["ID", "SIDE", "label", "p_tear"]].rename(columns={"p_tear": "zero_score"})
    sup = pd.read_csv(args.supervised_csv, dtype={"ID": str, "SIDE": str})
    sup = sup[["ID", "SIDE", "label", "prob_temperature_scaled"]].rename(
        columns={"prob_temperature_scaled": "supervised_score"}
    )
    merged = (fine.rename(columns={"score": "finetuned_score"})
              .merge(zero, on=["ID", "SIDE"], suffixes=("", "_zero"), validate="one_to_one")
              .merge(sup, on=["ID", "SIDE"], suffixes=("", "_sup"), validate="one_to_one"))
    if len(merged) != 566:
        raise ValueError(f"Expected all 566 locked test knees, found {len(merged)}")
    if not (merged["label"].eq(merged["label_zero"]).all()
            and merged["label"].eq(merged["label_sup"]).all()):
        raise ValueError("Label mismatch across paired prediction files")

    best_info = json.loads(info_path.read_text(encoding="utf-8"))
    threshold = float(best_info["validation"]["threshold"])
    y = merged["label"].to_numpy(dtype=int)
    score_names = ["finetuned_score", "zero_score", "supervised_score"]
    thresholds = {"finetuned_score": threshold, "zero_score": 0.5, "supervised_score": 0.5}
    point = {name: metric(y, merged[name].to_numpy(float), thresholds[name])
             for name in score_names}

    cluster_to_rows = {
        pid: np.flatnonzero(merged["ID"].to_numpy() == pid)
        for pid in merged["ID"].unique()
    }
    clusters = np.asarray(list(cluster_to_rows), dtype=object)
    rng = np.random.default_rng(args.seed)
    boot = {name: {key: [] for key in point[name]} for name in score_names}
    diffs = {
        "finetuned_minus_zero_auc": [],
        "finetuned_minus_supervised_auc": [],
    }
    arrays = {name: merged[name].to_numpy(float) for name in score_names}
    for _ in range(args.n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([cluster_to_rows[pid] for pid in sampled])
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        rep = {name: metric(yb, arrays[name][idx], thresholds[name]) for name in score_names}
        for name in score_names:
            for key, value in rep[name].items():
                boot[name][key].append(value)
        diffs["finetuned_minus_zero_auc"].append(rep["finetuned_score"]["auc"] - rep["zero_score"]["auc"])
        diffs["finetuned_minus_supervised_auc"].append(rep["finetuned_score"]["auc"] - rep["supervised_score"]["auc"])

    output = {
        "protocol": "locked OAI test; validation-selected fine-tuned threshold; participant-clustered bootstrap",
        "n_knees": int(len(merged)),
        "n_participants": int(len(clusters)),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "finetuned_validation_threshold": threshold,
        "metrics": {
            name: {key: {"estimate": value, "ci95": ci(boot[name][key])}
                   for key, value in point[name].items()}
            for name in score_names
        },
        "paired_auc_differences": {
            key: {"estimate": (
                point["finetuned_score"]["auc"] - point["zero_score"]["auc"]
                if "zero" in key else
                point["finetuned_score"]["auc"] - point["supervised_score"]["auc"]
            ), "ci95": ci(values)}
            for key, values in diffs.items()
        },
    }

    metrics_dir = ROOT / "results" / "metrics"
    logs_dir = ROOT / "results" / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(metrics_dir / f"{args.tag}_paired_predictions.csv", index=False)
    (metrics_dir / f"{args.tag}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    rows = []
    for model, values in output["metrics"].items():
        for name, result in values.items():
            rows.append({"model": model, "metric": name, **result})
    pd.DataFrame(rows).to_csv(metrics_dir / f"{args.tag}_metrics.csv", index=False)

    lines = ["# Locked-test 48-image VLM fine-tuning analysis", "",
             f"- Knees / participants: {len(merged)} / {len(clusters)}",
             f"- Validation-locked fine-tuned threshold: {threshold:.3f}",
             f"- Bootstrap: {args.n_boot:,} participant-clustered replicates", ""]
    for model, values in output["metrics"].items():
        auc = values["auc"]
        lines.append(f"- {model}: AUC {auc['estimate']:.3f} "
                     f"({auc['ci95'][0]:.3f}–{auc['ci95'][1]:.3f})")
    lines.extend(["", "## Paired AUC differences", ""])
    for name, result in output["paired_auc_differences"].items():
        lines.append(f"- {name}: {result['estimate']:+.3f} "
                     f"({result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f})")
    (logs_dir / f"{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
