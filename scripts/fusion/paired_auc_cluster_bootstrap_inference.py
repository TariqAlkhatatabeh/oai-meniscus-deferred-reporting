"""Paired participant-cluster bootstrap inference for fusion versus ConvNeXt AUC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    frame = pd.read_csv(args.predictions)
    required = {"ID", "label", "prob_convnext", "prob_fusion"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing values are not permitted")

    y = frame["label"].to_numpy(dtype=int)
    convnext = frame["prob_convnext"].to_numpy(dtype=float)
    fusion = frame["prob_fusion"].to_numpy(dtype=float)
    participant_ids = frame["ID"].drop_duplicates().to_numpy()
    frame_ids = frame["ID"].to_numpy()
    by_participant = {
        participant_id: np.flatnonzero(frame_ids == participant_id)
        for participant_id in participant_ids
    }

    observed = float(roc_auc_score(y, fusion) - roc_auc_score(y, convnext))
    rng = np.random.default_rng(args.seed)
    differences = np.empty(args.n_bootstrap, dtype=float)
    valid = 0

    while valid < args.n_bootstrap:
        sampled = rng.choice(participant_ids, len(participant_ids), replace=True)
        index = np.concatenate([by_participant[item] for item in sampled])
        boot_y = y[index]
        if np.unique(boot_y).size < 2:
            continue
        differences[valid] = float(
            roc_auc_score(boot_y, fusion[index])
            - roc_auc_score(boot_y, convnext[index])
        )
        valid += 1

    standard_error = float(np.std(differences, ddof=1))
    z_statistic = float(observed / standard_error)
    normal_p_value = float(math.erfc(abs(z_statistic) / math.sqrt(2.0)))
    centered = differences - observed
    extreme = int(np.sum(np.abs(centered) >= abs(observed) - 1e-15))
    centered_bootstrap_p = float((extreme + 1) / (args.n_bootstrap + 1))

    payload = {
        "comparison": "fusion_minus_convnext_auc",
        "null_hypothesis": "paired ROC AUC difference equals zero",
        "alternative": "two-sided",
        "resampling_unit": "participant; all knees from a sampled participant retained together",
        "n_knees": int(len(frame)),
        "n_participants": int(len(participant_ids)),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "observed_auc_difference": observed,
        "bootstrap_standard_error": standard_error,
        "bootstrap_percentile_95_ci": {
            "low": float(np.percentile(differences, 2.5)),
            "high": float(np.percentile(differences, 97.5)),
        },
        "two_sided_centered_bootstrap_p_value": centered_bootstrap_p,
        "centered_extreme_replicates": extreme,
        "two_sided_normal_p_value_from_bootstrap_se": normal_p_value,
        "z_statistic_from_bootstrap_se": z_statistic,
        "predictions": str(args.predictions.resolve()),
        "reporting_choice": (
            "Report the centered participant-cluster bootstrap p value with the "
            "paired percentile confidence interval."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
