"""Evaluate the primary ConvNeXt on MRNet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

import config as cfg
from dataset_multitask_16s_v2_aug import MeniscusDatasetMT16SV2
from eval_multitask_16s_v2_timm_exp_tta import (
    build_tta_pipeline,
    load_model_from_checkpoint,
    predict_with_tta,
)


DEFAULT_CKPT = (
    cfg.WEIGHTS
    / "exp_multitask_16s_v2_convnext_base_safe_strong_bs4_ckpt_20260628_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen transfer evaluation of the primary 16-slice ConvNeXt on MRNet."
    )
    parser.add_argument(
        "--index", type=Path, default=cfg.DATA / "mrnet_primary_16s_v1_index.csv"
    )
    parser.add_argument(
        "--splits", type=Path, default=cfg.DATA / "mrnet_da_splits.csv"
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tag", default="mrnet_primary_16s_frozen_v1")
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="fp16")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tta", action="store_true")
    return parser.parse_args()


def add_da_split(index: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    splits = pd.read_csv(split_path, dtype={"ID": str})[["ID", "da_split"]]
    if splits["ID"].duplicated().any():
        raise ValueError("Duplicate ID values in MRNet split file")
    merged = index.merge(splits, on="ID", how="left", validate="one_to_one")
    if merged["da_split"].isna().any():
        raise ValueError("Some MRNet examinations have no fixed DA split")
    return merged


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = y.astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": float(threshold),
    }


def bootstrap_intervals(
    y: np.ndarray, p: np.ndarray, n_boot: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    n = len(y)
    for _ in range(n_boot):
        ids = rng.integers(0, n, size=n)
        yb, pb = y[ids], p[ids]
        if np.unique(yb).size < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))
    return {
        "roc_auc_95ci": [float(v) for v in np.percentile(aucs, [2.5, 97.5])],
        "pr_auc_95ci": [float(v) for v in np.percentile(aps, [2.5, 97.5])],
        "valid_replicates": int(len(aucs)),
        "resampling_unit": "examination",
    }


def evaluate_partition(
    name: str,
    rows: pd.DataFrame,
    model: torch.nn.Module,
    device: str,
    tta_pipeline,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    eval_rows = rows.copy()
    eval_rows["severity"] = 0  # unused; the external endpoint is binary meniscus tear
    ds = MeniscusDatasetMT16SV2(eval_rows, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    probs, labels, _, _ = predict_with_tta(
        model,
        loader,
        device,
        tta_pipeline,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    metrics = point_metrics(labels, probs, threshold=0.5)
    metrics.update(bootstrap_intervals(labels, probs, args.bootstrap, args.seed))
    metrics["partition"] = name
    predictions = eval_rows[["ID", "split", "da_split", "label"]].copy()
    predictions["partition"] = name
    predictions["probability"] = probs
    return metrics, predictions


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    for path in (args.index, args.splits, args.ckpt):
        if not path.exists():
            raise FileNotFoundError(path)

    out_json = cfg.RESULTS / "metrics" / f"{args.tag}.json"
    out_csv = cfg.RESULTS / "metrics" / f"{args.tag}_predictions.csv"
    for path in (out_json, out_csv):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    amp_enabled = args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    model, ckpt, backbone, local_weights = load_model_from_checkpoint(args.ckpt, device)
    tta_pipeline = build_tta_pipeline(args.no_tta)

    index = pd.read_csv(args.index, dtype={"ID": str})
    index = add_da_split(index, args.splits)
    expected = [f"{seq}_{i:02d}" for seq in cfg.SEQUENCES for i in range(16)]
    missing = [col for col in expected if col not in index.columns]
    if missing:
        raise ValueError(f"Index is not a 16-slice tri-plane index; missing {missing[:3]}")

    partitions = {
        "official_valid": index[index["split"] == "valid"].copy(),
        "fixed_da_test": index[index["da_split"] == "test"].copy(),
    }
    all_metrics, all_predictions = {}, []
    for offset, (name, rows) in enumerate(partitions.items()):
        metrics, predictions = evaluate_partition(
            name,
            rows,
            model,
            device,
            tta_pipeline,
            amp_enabled,
            amp_dtype,
            argparse.Namespace(**{**vars(args), "seed": args.seed + offset}),
        )
        all_metrics[name] = metrics
        all_predictions.append(predictions)
        print(
            f"{name}: AUC {metrics['roc_auc']:.4f} "
            f"[{metrics['roc_auc_95ci'][0]:.4f}, {metrics['roc_auc_95ci'][1]:.4f}], "
            f"PR-AUC {metrics['pr_auc']:.4f}, n={metrics['n']}"
        )

    payload = {
        "analysis": "exact primary 16-slice ConvNeXt frozen MRNet transfer",
        "checkpoint": str(args.ckpt),
        "backbone": backbone,
        "local_initialization": str(local_weights),
        "checkpoint_best_validation_auc": ckpt.get("val_metrics", {}).get("auc"),
        "mrnet_index": str(args.index),
        "fixed_split_file": str(args.splits),
        "n_slices_per_plane": 16,
        "tta": [name for name, _ in tta_pipeline],
        "threshold_source": "unchanged primary-model operating point (0.5)",
        "label_use": "metrics only; no MRNet fitting, calibration, or threshold selection",
        "metrics": all_metrics,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.concat(all_predictions, ignore_index=True).to_csv(out_csv, index=False)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
