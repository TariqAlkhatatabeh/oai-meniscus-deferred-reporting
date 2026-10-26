"""Routing, coverage and subset-metric helpers."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

def make_prediction_frame(rows, probs, labels, threshold: float):
    df = pd.DataFrame({
        "ID": rows["ID"].astype(str).values,
        "SIDE": rows["SIDE"].values,
        "label": labels.astype(int),
        "prob": probs.astype(float),
    })
    df["pred_0p5"] = (df["prob"] >= threshold).astype(int)
    df["confidence"] = np.maximum(df["prob"], 1.0 - df["prob"])
    df["uncertainty"] = 1.0 - df["confidence"]
    df["abs_margin_from_0p5"] = np.abs(df["prob"] - threshold)
    df["correct_0p5"] = (df["pred_0p5"] == df["label"]).astype(int)
    return df

def safe_auc(labels, scores):
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))

def safe_pr_auc(labels, scores):
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))

def metrics_for_subset(df, threshold: float):
    y = df["label"].astype(int).to_numpy()
    p = df["prob"].astype(float).to_numpy()
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(df)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": safe_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "error_rate": float(1.0 - accuracy_score(y, pred)) if len(y) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred))
        if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "confusion": {"tp": int(tp), "fp": int(fp),
                      "fn": int(fn), "tn": int(tn)},
    }

def cutoff_for_target_coverage(val_df, target_coverage: float):
    conf = np.sort(val_df["confidence"].astype(float).to_numpy())[::-1]
    n_keep = max(1, min(len(conf), int(math.ceil(target_coverage * len(conf)))))
    return float(conf[n_keep - 1])

def coverage_result(split_name, df, cutoff: float, threshold: float):
    covered = df[df["confidence"] >= cutoff].copy()
    deferred = df[df["confidence"] < cutoff].copy()
    return {
        "split": split_name,
        "confidence_cutoff": float(cutoff),
        "n_total": int(len(df)),
        "n_covered": int(len(covered)),
        "n_deferred": int(len(deferred)),
        "coverage": float(len(covered) / len(df)) if len(df) else float("nan"),
        "defer_rate": float(len(deferred) / len(df)) if len(df) else float("nan"),
        "covered": metrics_for_subset(covered, threshold),
        "deferred": metrics_for_subset(deferred, threshold) if len(deferred) else None,
    }

def plot_coverage(path, rows, full_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [row["target_coverage"] for row in rows]
    test_cov = [row["test"]["coverage"] for row in rows]
    auc = [row["test"]["covered"]["auc"] for row in rows]
    f1 = [row["test"]["covered"]["f1"] for row in rows]
    error = [row["test"]["covered"]["error_rate"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax1.plot(test_cov, auc, marker="o", label="AUC")
    ax1.plot(test_cov, f1, marker="s", label="F1")
    ax1.axhline(full_metrics["auc"], linestyle=":", color="C0", alpha=0.5)
    ax1.axhline(full_metrics["f1"], linestyle=":", color="C1", alpha=0.5)
    ax1.set_xlabel("Actual Test Coverage")
    ax1.set_ylabel("Metric")
    ax1.set_ylim(0.70, 1.00)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(test_cov, error, marker="^", color="C3", label="Error rate")
    ax2.set_ylabel("Error Rate")
    ax2.set_ylim(0.0, max(0.25, max(error) * 1.25))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left")
    ax1.set_title("Validation-Locked Confidence Coverage")
    ax1.set_xticks(test_cov)
    ax1.set_xticklabels([f"{value:.0%}" for value in test_cov])
    for x_value, target in zip(test_cov, targets):
        ax1.annotate(f"val {target:.0%}", (x_value, 0.985),
                     ha="center", va="top", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
