"""Export per-knee severity probabilities from the locked primary 16-slice model.

This evaluation-only script reuses the same seven-view TTA implementation used
for the primary binary model. It writes a new, versioned CSV and never modifies
the locked split or existing prediction artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

import config as cfg  # noqa: E402
from dataset_multitask_16s_v2_aug import (  # noqa: E402
    MeniscusDatasetMT16SV2,
    load_mt_split,
    verify_no_leakage_mt,
)
from eval_multitask_16s_v2_timm_exp_tta import (  # noqa: E402
    build_tta_pipeline,
    load_model_from_checkpoint,
    predict_with_tta,
    severity_metrics_from_probs,
)

DEFAULT_CKPT = (
    cfg.WEIGHTS
    / "exp_multitask_16s_v2_convnext_base_safe_strong_bs4_ckpt_20260628_best.pt"
)
DEFAULT_OUT = (
    PROJECT
    / "results"
    / "reporting"
    / "primary16s_test_severity_tta7.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", choices=["fp16", "bf16", "off"], default="fp16")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    verify_no_leakage_mt()
    rows = load_mt_split("test")
    dataset = MeniscusDatasetMT16SV2(rows, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    model, checkpoint, backbone, local_weights = load_model_from_checkpoint(
        args.ckpt, "cuda"
    )
    tta = build_tta_pipeline(no_tta=False)
    amp_enabled = args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    main_prob, labels, severity_prob, severity_labels = predict_with_tta(
        model,
        loader,
        "cuda",
        tta,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        max_batches=args.max_batches,
    )

    n = len(main_prob)
    source_rows = rows.iloc[:n].reset_index(drop=True)
    output = pd.DataFrame(
        {
            "ID": source_rows["ID"].astype(str),
            "SIDE": source_rows["SIDE"].astype(int),
            "label": labels.astype(int),
            "binary_prob_recomputed": main_prob,
            "severity_label": severity_labels.astype(int),
            "severity_prob_grade0": severity_prob[:, 0],
            "severity_prob_grade1": severity_prob[:, 1],
            "severity_prob_grade2plus": severity_prob[:, 2],
            "severity_pred": severity_prob.argmax(axis=1).astype(int),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    metadata = {
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "backbone": backbone,
        "local_weights": str(local_weights),
        "split": "locked OAI test",
        "n": len(output),
        "seed": int(cfg.SEED),
        "tta": [name for name, _ in tta],
        "severity_metrics": severity_metrics_from_probs(severity_prob, severity_labels),
        "note": "Severity output is from the primary 16-slice ConvNeXt auxiliary head.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
