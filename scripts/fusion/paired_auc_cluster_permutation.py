"""Participant-clustered paired permutation test for fusion versus ConvNeXt AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=50_000)
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
    if frame[["ID", "label", "prob_convnext", "prob_fusion"]].isna().any().any():
        raise ValueError("Missing values are not permitted")

    y = frame["label"].to_numpy(dtype=int)
    convnext = frame["prob_convnext"].to_numpy(dtype=float)
    fusion = frame["prob_fusion"].to_numpy(dtype=float)
    participant_codes, participant_ids = pd.factorize(frame["ID"], sort=False)
    n_participants = len(participant_ids)

    observed = float(roc_auc_score(y, fusion) - roc_auc_score(y, convnext))
    rng = np.random.default_rng(args.seed)
    extreme = 0
    permutation_differences = np.empty(args.n_permutations, dtype=float)

    for index in range(args.n_permutations):
        swap_by_participant = rng.integers(0, 2, size=n_participants, dtype=np.int8)
        swap = swap_by_participant[participant_codes].astype(bool)
        perm_fusion = np.where(swap, convnext, fusion)
        perm_convnext = np.where(swap, fusion, convnext)
        difference = float(
            roc_auc_score(y, perm_fusion) - roc_auc_score(y, perm_convnext)
        )
        permutation_differences[index] = difference
        if abs(difference) >= abs(observed) - 1e-15:
            extreme += 1

    p_value = float((extreme + 1) / (args.n_permutations + 1))
    monte_carlo_se = float(
        np.sqrt(p_value * (1.0 - p_value) / (args.n_permutations + 1))
    )
    payload = {
        "comparison": "fusion_minus_convnext_auc",
        "null_hypothesis": "equal ROC AUC for the two paired score vectors",
        "alternative": "two-sided",
        "permutation_unit": "participant; all knees from a participant swapped together",
        "n_knees": int(len(frame)),
        "n_participants": int(n_participants),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "observed_auc_difference": observed,
        "extreme_permutations": int(extreme),
        "p_value_plus_one_correction": p_value,
        "monte_carlo_standard_error": monte_carlo_se,
        "permutation_null_quantiles": {
            "2.5%": float(np.percentile(permutation_differences, 2.5)),
            "50%": float(np.percentile(permutation_differences, 50.0)),
            "97.5%": float(np.percentile(permutation_differences, 97.5)),
        },
        "predictions": str(args.predictions.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
