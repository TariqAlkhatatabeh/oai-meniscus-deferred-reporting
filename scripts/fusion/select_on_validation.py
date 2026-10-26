"""Select and freeze ConvNeXt/VLM late fusion using validation data only.

This program intentionally has no test-set argument. It produces the immutable
protocol consumed by evaluate_locked_test.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_core import (align, apply_linear_logistic, best_f1_threshold,
                         json_dump, load_predictions, logit, metrics, sha256,
                         stable_key_hash)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convnext-val", type=Path, required=True)
    parser.add_argument("--vlm-val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-knees", type=int, default=540)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_linear_logistic(x: np.ndarray, y: np.ndarray) -> tuple[dict, LogisticRegression, StandardScaler]:
    features = np.asarray(x, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                               random_state=42).fit(scaler.transform(features), y)
    params = {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coef": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
    }
    return params, model, scaler


def predict_model(model: LogisticRegression, scaler: StandardScaler,
                  x: np.ndarray) -> np.ndarray:
    features = np.asarray(x, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    return model.predict_proba(scaler.transform(features))[:, 1]


def main() -> None:
    args = parse_args()
    protocol_path = args.out_dir / "frozen_fusion_protocol.json"
    if protocol_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite frozen protocol: {protocol_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conv = load_predictions(args.convnext_val, "prob")
    vlm = load_predictions(args.vlm_val, "score")
    frame = align(conv, vlm)
    if len(frame) != args.expected_knees:
        raise ValueError(f"Expected {args.expected_knees} knees, found {len(frame)}")
    y = frame["label"].to_numpy(int)
    groups = frame["ID"].to_numpy(str)
    raw = np.column_stack([
        logit(frame["prob_convnext"].to_numpy(float)),
        logit(frame["prob_vlm"].to_numpy(float)),
    ])

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                    random_state=args.seed)
    oof_conv = np.full(len(frame), np.nan)
    oof_vlm = np.full(len(frame), np.nan)
    oof_stack = np.full(len(frame), np.nan)
    for train_index, held_index in splitter.split(raw, y, groups):
        conv_params, conv_model, conv_scaler = fit_linear_logistic(
            raw[train_index, 0], y[train_index])
        vlm_params, vlm_model, vlm_scaler = fit_linear_logistic(
            raw[train_index, 1], y[train_index])
        _, stack_model, stack_scaler = fit_linear_logistic(raw[train_index], y[train_index])
        oof_conv[held_index] = predict_model(conv_model, conv_scaler, raw[held_index, 0])
        oof_vlm[held_index] = predict_model(vlm_model, vlm_scaler, raw[held_index, 1])
        oof_stack[held_index] = predict_model(stack_model, stack_scaler, raw[held_index])
    if not np.isfinite(np.c_[oof_conv, oof_vlm, oof_stack]).all():
        raise RuntimeError("Incomplete out-of-fold predictions")

    final_conv, _, _ = fit_linear_logistic(raw[:, 0], y)
    final_vlm, _, _ = fit_linear_logistic(raw[:, 1], y)
    final_stack, _, _ = fit_linear_logistic(raw, y)

    candidate_probabilities: dict[str, np.ndarray] = {
        "equal_platt_blend": 0.5 * oof_conv + 0.5 * oof_vlm,
        "logistic_stack": oof_stack,
    }
    candidate_definitions: dict[str, dict] = {
        "equal_platt_blend": {"type": "weighted_platt_blend", "convnext_weight": 0.5},
        "logistic_stack": {"type": "logistic_stack"},
    }
    for weight in (0.6, 0.7, 0.8, 0.9):
        name = f"weighted_platt_blend_conv{int(weight * 100)}"
        candidate_probabilities[name] = weight * oof_conv + (1 - weight) * oof_vlm
        candidate_definitions[name] = {
            "type": "weighted_platt_blend", "convnext_weight": weight}

    baseline = {}
    for name, probability in {"convnext": oof_conv, "vlm": oof_vlm}.items():
        threshold = best_f1_threshold(y, probability)
        baseline[name] = metrics(y, probability, threshold)

    candidates = {}
    for name, probability in candidate_probabilities.items():
        threshold = best_f1_threshold(y, probability)
        record = metrics(y, probability, threshold)
        record.update(candidate_definitions[name])
        record["eligible"] = True
        if name == "logistic_stack" and any(value < 0 for value in final_stack["coef"]):
            record["eligible"] = False
            record["ineligible_reason"] = "negative final stacking coefficient"
        candidates[name] = record

    eligible = [name for name, record in candidates.items() if record["eligible"]]
    selected_name = max(
        eligible,
        key=lambda name: (candidates[name]["auc"], candidates[name]["pr_auc"],
                          -candidates[name]["brier"]),
    )
    selected = candidates[selected_name]
    selected_probability = candidate_probabilities[selected_name]

    frame["oof_convnext_calibrated"] = oof_conv
    frame["oof_vlm_calibrated"] = oof_vlm
    frame["oof_selected_fusion"] = selected_probability
    frame.to_csv(args.out_dir / "validation_oof_predictions.csv", index=False)

    protocol = {
        "protocol_version": "convnext_vlm_late_fusion_v1",
        "selection_boundary": "validation_only_grouped_5_fold_oof",
        "seed": args.seed,
        "folds": args.folds,
        "n_knees": len(frame),
        "n_participants": int(frame["ID"].nunique()),
        "n_positive_knees": int(y.sum()),
        "validation_key_sha256": stable_key_hash(frame),
        "validation_inputs": {
            "convnext": {"path": str(args.convnext_val.resolve()), "sha256": sha256(args.convnext_val)},
            "vlm": {"path": str(args.vlm_val.resolve()), "sha256": sha256(args.vlm_val)},
        },
        "raw_features": ["logit_convnext_probability", "logit_vlm_score"],
        "base_calibrators_fit_on_all_validation": {
            "convnext": final_conv, "vlm": final_vlm},
        "stacker_fit_on_all_validation": final_stack,
        "baseline_oof_metrics": baseline,
        "candidate_oof_metrics": candidates,
        "selected_candidate": selected_name,
        "selected_definition": candidate_definitions[selected_name],
        "selected_validation_threshold": selected["threshold"],
        "standalone_validation_thresholds": {
            "convnext_raw": best_f1_threshold(y, frame["prob_convnext"].to_numpy(float)),
            "vlm_raw": best_f1_threshold(y, frame["prob_vlm"].to_numpy(float)),
        },
        "selection_rule": "maximum OOF ROC-AUC; PR-AUC then lower Brier as tie-breakers",
        "test_data_seen": False,
    }
    json_dump(protocol_path, protocol)

    lines = [
        "# ConvNeXt-VLM fusion validation selection", "",
        "This artifact was produced without a test-set input.", "",
        f"- Selected candidate: `{selected_name}`",
        f"- Validation knees/participants: {len(frame)}/{frame['ID'].nunique()}",
        f"- Selected OOF AUC: {selected['auc']:.4f}",
        f"- ConvNeXt OOF AUC: {baseline['convnext']['auc']:.4f}",
        f"- VLM OOF AUC: {baseline['vlm']['auc']:.4f}", "",
        "| Candidate | Eligible | AUC | PR-AUC | F1 | Brier |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in candidates.items():
        lines.append(f"| {name} | {record['eligible']} | {record['auc']:.4f} | "
                     f"{record['pr_auc']:.4f} | {record['f1']:.4f} | {record['brier']:.4f} |")
    (args.out_dir / "validation_selection_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Frozen validation-only fusion protocol: {protocol_path}")


if __name__ == "__main__":
    main()

