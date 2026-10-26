"""Paired comparison of frozen and adapted MRNet scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen",
        type=Path,
        default=cfg.RESULTS / "metrics" / "mrnet_primary_16s_frozen_v1_predictions.csv",
    )
    parser.add_argument(
        "--adapted",
        type=Path,
        default=cfg.RESULTS / "metrics" / "mrnet_primary_16s_da_v2_resumable_test_predictions.csv",
    )
    parser.add_argument("--tag", default="mrnet_primary_16s_zero_vs_da_v3_final")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def interval(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def main():
    args = parse_args()
    frozen = pd.read_csv(args.frozen, dtype={"ID": str})
    frozen = frozen[frozen.partition == "fixed_da_test"].copy()
    adapted = pd.read_csv(args.adapted, dtype={"ID": str})
    joined = frozen[["ID", "label", "probability"]].merge(
        adapted[["ID", "label", "probability"]],
        on="ID",
        suffixes=("_frozen", "_adapted"),
        validate="one_to_one",
    )
    if len(joined) != 188:
        raise ValueError(f"Expected 188 paired examinations, found {len(joined)}")
    if not np.array_equal(joined.label_frozen, joined.label_adapted):
        raise ValueError("Label mismatch between paired prediction files")

    y = joined.label_frozen.to_numpy(int)
    p0 = joined.probability_frozen.to_numpy(float)
    p1 = joined.probability_adapted.to_numpy(float)
    rng = np.random.default_rng(args.seed)
    auc0, auc1, ap0, ap1 = [], [], [], []
    for _ in range(args.bootstrap):
        ids = rng.integers(0, len(y), size=len(y))
        yb = y[ids]
        if np.unique(yb).size < 2:
            continue
        auc0.append(roc_auc_score(yb, p0[ids]))
        auc1.append(roc_auc_score(yb, p1[ids]))
        ap0.append(average_precision_score(yb, p0[ids]))
        ap1.append(average_precision_score(yb, p1[ids]))
    auc0, auc1 = np.asarray(auc0), np.asarray(auc1)
    ap0, ap1 = np.asarray(ap0), np.asarray(ap1)
    auc_diff = auc1 - auc0
    ap_diff = ap1 - ap0
    # Add-one correction avoids an impossible p=0 when no bootstrap replicate
    # crosses the null; resolution is determined by the number of replicates.
    tail_count = min(int(np.sum(auc_diff <= 0)), int(np.sum(auc_diff >= 0)))
    p_two_sided = float(min(1.0, 2 * (tail_count + 1) / (len(auc_diff) + 1)))
    payload = {
        "comparison": "same primary 16-slice ConvNeXt, MRNet adapted versus frozen",
        "n": int(len(y)),
        "bootstrap_replicates": int(len(auc_diff)),
        "resampling_unit": "examination",
        "frozen": {
            "roc_auc": float(roc_auc_score(y, p0)),
            "roc_auc_95ci": interval(auc0),
            "pr_auc": float(average_precision_score(y, p0)),
            "pr_auc_95ci": interval(ap0),
        },
        "adapted": {
            "roc_auc": float(roc_auc_score(y, p1)),
            "roc_auc_95ci": interval(auc1),
            "pr_auc": float(average_precision_score(y, p1)),
            "pr_auc_95ci": interval(ap1),
        },
        "paired_difference_adapted_minus_frozen": {
            "roc_auc": float(roc_auc_score(y, p1) - roc_auc_score(y, p0)),
            "roc_auc_95ci": interval(auc_diff),
            "roc_auc_bootstrap_p_two_sided": p_two_sided,
            "pr_auc": float(average_precision_score(y, p1) - average_precision_score(y, p0)),
            "pr_auc_95ci": interval(ap_diff),
        },
    }
    out_json = cfg.RESULTS / "metrics" / f"{args.tag}.json"
    out_csv = cfg.RESULTS / "metrics" / f"{args.tag}_paired_predictions.csv"
    for path in (out_json, out_csv):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite: {path}")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joined.to_csv(out_csv, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
