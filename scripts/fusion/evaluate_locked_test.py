"""Evaluate exactly one validation-frozen ConvNeXt/VLM fusion on locked test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_core import (align, apply_linear_logistic, json_dump,
                         load_predictions, logit, metrics, sha256)


UNLOCK = "I_CONFIRM_FUSION_IS_FROZEN"
METRIC_NAMES = ["auc", "pr_auc", "f1", "accuracy", "sensitivity",
                "specificity", "brier", "ece_10bin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--convnext-test", type=Path, required=True)
    parser.add_argument("--vlm-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlock-locked-test", required=True)
    parser.add_argument("--expected-knees", type=int, default=566)
    return parser.parse_args()


def cluster_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    ids = frame["ID"].drop_duplicates().to_numpy()
    by_id = {pid: np.flatnonzero(frame["ID"].to_numpy() == pid) for pid in ids}
    sampled = rng.choice(ids, len(ids), replace=True)
    return np.concatenate([by_id[pid] for pid in sampled])


def ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {"low": float(np.percentile(array, 2.5)),
            "high": float(np.percentile(array, 97.5)),
            "valid_replicates": int(len(array))}


def main() -> None:
    args = parse_args()
    if args.unlock_locked_test != UNLOCK:
        raise SystemExit(f"Locked-test guard refused. Required phrase: {UNLOCK}")
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.out_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("test_data_seen") is not False:
        raise ValueError("Protocol does not certify a validation-only selection boundary")
    if protocol.get("protocol_version") != "convnext_vlm_late_fusion_v1":
        raise ValueError("Unexpected protocol version")
    for record in protocol["validation_inputs"].values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Validation input changed after freezing: {path}")

    conv = load_predictions(args.convnext_test, "prob")
    vlm = load_predictions(args.vlm_test, "score")
    frame = align(conv, vlm)
    if len(frame) != args.expected_knees:
        raise ValueError(f"Expected {args.expected_knees} test knees, found {len(frame)}")
    val_ids = set()
    for record in protocol["validation_inputs"].values():
        val_ids.update(load_predictions(Path(record["path"]))["ID"].astype(str))
    overlap = set(frame["ID"].astype(str)).intersection(val_ids)
    if overlap:
        raise ValueError(f"Participant leakage between validation and test: {sorted(overlap)[:5]}")

    y = frame["label"].to_numpy(int)
    raw = np.column_stack([logit(frame["prob_convnext"]), logit(frame["prob_vlm"])])
    calibrated_conv = apply_linear_logistic(
        raw[:, 0], protocol["base_calibrators_fit_on_all_validation"]["convnext"])
    calibrated_vlm = apply_linear_logistic(
        raw[:, 1], protocol["base_calibrators_fit_on_all_validation"]["vlm"])
    definition = protocol["selected_definition"]
    if definition["type"] == "weighted_platt_blend":
        weight = float(definition["convnext_weight"])
        fused = weight * calibrated_conv + (1 - weight) * calibrated_vlm
    elif definition["type"] == "logistic_stack":
        fused = apply_linear_logistic(raw, protocol["stacker_fit_on_all_validation"])
    else:
        raise ValueError(f"Unknown frozen fusion type: {definition['type']}")

    probabilities = {
        "convnext": frame["prob_convnext"].to_numpy(float),
        "vlm": frame["prob_vlm"].to_numpy(float),
        "fusion": fused,
    }
    thresholds = {
        "convnext": float(protocol["standalone_validation_thresholds"]["convnext_raw"]),
        "vlm": float(protocol["standalone_validation_thresholds"]["vlm_raw"]),
        "fusion": float(protocol["selected_validation_threshold"]),
    }
    point = {name: metrics(y, probability, thresholds[name])
             for name, probability in probabilities.items()}

    rng = np.random.default_rng(args.seed)
    samples = {name: {metric: [] for metric in METRIC_NAMES}
               for name in probabilities}
    differences = {comparison: {metric: [] for metric in METRIC_NAMES}
                   for comparison in ("fusion_minus_convnext", "fusion_minus_vlm")}
    for _ in range(args.n_bootstrap):
        index = cluster_indices(frame, rng)
        boot_y = y[index]
        if len(np.unique(boot_y)) < 2:
            continue
        boot_metrics = {
            name: metrics(boot_y, probability[index], thresholds[name])
            for name, probability in probabilities.items()}
        for name in probabilities:
            for metric in METRIC_NAMES:
                samples[name][metric].append(boot_metrics[name][metric])
        for metric in METRIC_NAMES:
            differences["fusion_minus_convnext"][metric].append(
                boot_metrics["fusion"][metric] - boot_metrics["convnext"][metric])
            differences["fusion_minus_vlm"][metric].append(
                boot_metrics["fusion"][metric] - boot_metrics["vlm"][metric])

    intervals = {name: {metric: ci(values) for metric, values in block.items()}
                 for name, block in samples.items()}
    difference_point = {
        "fusion_minus_convnext": {metric: point["fusion"][metric] - point["convnext"][metric]
                                  for metric in METRIC_NAMES},
        "fusion_minus_vlm": {metric: point["fusion"][metric] - point["vlm"][metric]
                              for metric in METRIC_NAMES},
    }
    difference_intervals = {
        name: {metric: ci(values) for metric, values in block.items()}
        for name, block in differences.items()}

    args.out_dir.mkdir(parents=True, exist_ok=False)
    output = frame.copy()
    output["prob_convnext_calibrated"] = calibrated_conv
    output["prob_vlm_calibrated"] = calibrated_vlm
    output["prob_fusion"] = fused
    output.to_csv(args.out_dir / "paired_test_predictions.csv", index=False)
    payload = {
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "selected_candidate": protocol["selected_candidate"],
        "n_knees": len(frame), "n_participants": int(frame["ID"].nunique()),
        "n_positive_knees": int(y.sum()), "n_bootstrap": args.n_bootstrap,
        "point_metrics": point, "confidence_intervals": intervals,
        "paired_differences": difference_point,
        "paired_difference_confidence_intervals": difference_intervals,
        "test_inputs": {
            "convnext": {"path": str(args.convnext_test.resolve()), "sha256": sha256(args.convnext_test)},
            "vlm": {"path": str(args.vlm_test.resolve()), "sha256": sha256(args.vlm_test)},
        },
    }
    json_dump(args.out_dir / "locked_test_results.json", payload)
    lines = ["# Locked-test ConvNeXt-VLM fusion evaluation", "",
             f"- Frozen candidate: `{protocol['selected_candidate']}`",
             f"- Knees/participants: {len(frame)}/{frame['ID'].nunique()}",
             f"- Participant-clustered bootstrap replicates: {args.n_bootstrap}", "",
             "| Model | AUC (95% CI) | PR-AUC (95% CI) | F1 | Brier |", "|---|---:|---:|---:|---:|"]
    for name in ("convnext", "vlm", "fusion"):
        auc_ci = intervals[name]["auc"]
        pr_ci = intervals[name]["pr_auc"]
        lines.append(f"| {name} | {point[name]['auc']:.4f} ({auc_ci['low']:.4f}-{auc_ci['high']:.4f}) | "
                     f"{point[name]['pr_auc']:.4f} ({pr_ci['low']:.4f}-{pr_ci['high']:.4f}) | "
                     f"{point[name]['f1']:.4f} | {point[name]['brier']:.4f} |")
    lines.extend(["", "## Paired fusion gains", "",
                  "| Comparison | AUC difference (95% CI) | PR-AUC difference (95% CI) |",
                  "|---|---:|---:|"])
    for comparison in ("fusion_minus_convnext", "fusion_minus_vlm"):
        auc_ci = difference_intervals[comparison]["auc"]
        pr_ci = difference_intervals[comparison]["pr_auc"]
        lines.append(f"| {comparison} | {difference_point[comparison]['auc']:+.4f} "
                     f"({auc_ci['low']:+.4f}-{auc_ci['high']:+.4f}) | "
                     f"{difference_point[comparison]['pr_auc']:+.4f} "
                     f"({pr_ci['low']:+.4f}-{pr_ci['high']:+.4f}) |")
    (args.out_dir / "locked_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "locked_test_report.md")


if __name__ == "__main__":
    main()
