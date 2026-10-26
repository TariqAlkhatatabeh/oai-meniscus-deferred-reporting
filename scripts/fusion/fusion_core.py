"""Shared, CPU-only utilities for the prespecified ConvNeXt/VLM fusion study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


KEYS = ["ID", "SIDE"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key_hash(frame: pd.DataFrame) -> str:
    keys = frame[KEYS].astype(str).agg("_".join, axis=1).sort_values()
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def load_predictions(path: Path, preferred_score: str | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "SIDE", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    score_col = preferred_score
    if score_col is None:
        score_col = "prob" if "prob" in frame.columns else "score"
    if score_col not in frame.columns:
        raise ValueError(f"{path} has no requested score column {score_col!r}")
    out = frame[["ID", "SIDE", "label", score_col]].rename(
        columns={score_col: "prob"}).copy()
    out["ID"] = out["ID"].astype(str)
    out["SIDE"] = out["SIDE"].astype(int)
    out["label"] = out["label"].astype(int)
    out["prob"] = out["prob"].astype(float)
    if out.duplicated(KEYS).any():
        raise ValueError(f"Duplicate knee keys in {path}")
    if not out["label"].isin([0, 1]).all():
        raise ValueError(f"Non-binary labels in {path}")
    if not np.isfinite(out["prob"]).all():
        raise ValueError(f"Non-finite scores in {path}")
    if ((out["prob"] < 0) | (out["prob"] > 1)).any():
        raise ValueError(f"Scores outside [0,1] in {path}")
    return out.sort_values(KEYS).reset_index(drop=True)


def align(convnext: pd.DataFrame, vlm: pd.DataFrame) -> pd.DataFrame:
    left = convnext.rename(columns={"prob": "prob_convnext"})
    right = vlm.rename(columns={"prob": "prob_vlm", "label": "label_vlm"})
    merged = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"Prediction keys do not align: ConvNeXt={len(left)}, "
            f"VLM={len(right)}, merged={len(merged)}")
    if not np.array_equal(merged["label"].to_numpy(), merged["label_vlm"].to_numpy()):
        raise ValueError("Label mismatch between ConvNeXt and VLM predictions")
    return merged.drop(columns="label_vlm").sort_values(KEYS).reset_index(drop=True)


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-value))


def best_f1_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    candidates = np.unique(np.r_[0.0, p, 1.0])
    scored = [(f1_score(y, p >= threshold, zero_division=0), threshold)
              for threshold in candidates]
    best = max(scored, key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))
    return float(best[1])


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray,
                               bins: int = 10) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y, p, bins=10),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def apply_linear_logistic(raw_features: np.ndarray, parameters: dict) -> np.ndarray:
    x = np.asarray(raw_features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mean = np.asarray(parameters["mean"], dtype=float)
    scale = np.asarray(parameters["scale"], dtype=float)
    coef = np.asarray(parameters["coef"], dtype=float)
    standardized = (x - mean) / scale
    return sigmoid(standardized @ coef + float(parameters["intercept"]))


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

