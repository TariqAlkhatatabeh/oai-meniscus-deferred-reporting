"""Seven-view TTA evaluation for the primary model."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg


def _move_x_to_device(x, device):
    return {key: value.to(device, non_blocking=True) for key, value in x.items()}


def _identity(x):
    return x


def _rotate_stack_batch(stack: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    batch, n_slices, channels, height, width = stack.shape
    flat = stack.reshape(batch * n_slices, channels, height, width)
    angle = torch.tensor(
        np.deg2rad(angle_degrees), dtype=flat.dtype, device=flat.device)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    theta = torch.stack([
        torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)]),
        torch.stack([sin_a, cos_a, torch.zeros_like(cos_a)]),
    ])
    theta = theta.unsqueeze(0).expand(flat.size(0), -1, -1)
    grid = F.affine_grid(theta, flat.size(), align_corners=False)
    out = F.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros",
        align_corners=False)
    return out.reshape(batch, n_slices, channels, height, width)


def _rotate(x, angle_degrees: float):
    return {key: _rotate_stack_batch(value, angle_degrees)
            for key, value in x.items()}


def _gamma(x, gamma: float):
    return {key: torch.clamp(value ** gamma, 0.0, 1.0)
            for key, value in x.items()}


def _noise(x, sigma: float):
    return {key: torch.clamp(value + torch.randn_like(value) * sigma, 0.0, 1.0)
            for key, value in x.items()}


def build_tta_pipeline(no_tta: bool):
    if no_tta:
        return [("original", _identity)]
    return [
        ("original", _identity),
        ("rot+5", lambda x: _rotate(x, 5.0)),
        ("rot-5", lambda x: _rotate(x, -5.0)),
        ("gamma0.9", lambda x: _gamma(x, 0.9)),
        ("gamma1.1", lambda x: _gamma(x, 1.1)),
        ("noise0.01", lambda x: _noise(x, 0.01)),
        ("noise0.02", lambda x: _noise(x, 0.02)),
    ]


def binary_metrics_from_probs(probs, labels, threshold=0.5):
    labels_int = labels.astype(int)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_int, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels_int, probs))
        if len(np.unique(labels_int)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels_int, probs)),
        "bal_acc": float(balanced_accuracy_score(labels_int, preds)),
        "f1": float(f1_score(labels_int, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion": {"tp": int(tp), "fp": int(fp),
                      "fn": int(fn), "tn": int(tn)},
    }


def severity_metrics_from_probs(probs, labels):
    preds = probs.argmax(axis=1)
    return {
        "sev_acc": float(accuracy_score(labels.astype(int), preds)),
        "sev_f1_macro": float(f1_score(
            labels.astype(int), preds, average="macro", zero_division=0)),
    }


def find_best_threshold(labels, probs, metric="f1", n_steps=200):
    labels = labels.astype(int)
    best_thr, best_val = 0.5, -1.0
    sweep = []
    for thr in np.linspace(0.01, 0.99, n_steps):
        preds = (probs >= thr).astype(int)
        if metric == "f1":
            value = float(f1_score(labels, preds, zero_division=0))
        elif metric == "bal_acc":
            value = float(balanced_accuracy_score(labels, preds))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        sweep.append({"threshold": float(thr), metric: value})
        if value > best_val:
            best_val = value
            best_thr = float(thr)
    return best_thr, best_val, sweep


@torch.no_grad()
def predict_with_tta(model, loader, device, tta_pipeline,
                     amp_enabled=False, amp_dtype=torch.float16,
                     max_batches=0):
    model.eval()
    all_main_probs, all_y = [], []
    all_sev_probs, all_sev_y = [], []

    for batch_idx, (x, y, severity) in enumerate(
            tqdm(loader, desc="TTA eval", leave=False)):
        x_dev = _move_x_to_device(x, device)
        n_tta = len(tta_pipeline)
        main_sum = torch.zeros(y.size(0), device=device)
        sev_sum = torch.zeros(y.size(0), 3, device=device)

        for _name, fn in tta_pipeline:
            x_aug = fn(x_dev)
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_enabled):
                main_logits, sev_logits = model(x_aug)
            main_sum += torch.sigmoid(main_logits.squeeze(-1).float())
            sev_sum += torch.softmax(sev_logits.float(), dim=1)

        all_main_probs.append((main_sum / n_tta).cpu().numpy())
        all_sev_probs.append((sev_sum / n_tta).cpu().numpy())
        all_y.append(y.float().numpy())
        all_sev_y.append(severity.long().numpy())

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    return (
        np.concatenate(all_main_probs),
        np.concatenate(all_y),
        np.concatenate(all_sev_probs),
        np.concatenate(all_sev_y),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimental timm-backbone 16s TTA evaluation")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--metric", choices=["f1", "bal_acc"], default="f1")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"],
                        default="fp16")
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--max_test_batches", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path: Path, device: str):
    from model_multitask_16s_v2_timm_exp import TriPlaneMultiTask16SV2Timm

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt.get("backbone") or ckpt_args.get("backbone")
    local_weights = ckpt.get("local_weights") or ckpt_args.get("local_weights")
    if backbone is None or local_weights is None:
        raise ValueError(
            f"{ckpt_path} does not contain backbone/local_weights metadata")
    model = TriPlaneMultiTask16SV2Timm(
        backbone,
        Path(local_weights),
        dropout=float(ckpt_args.get("dropout", ckpt.get("dropout", 0.3))),
        grad_checkpointing=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, backbone, Path(local_weights)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    amp_enabled = args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    import pandas as pd
    from dataset_multitask_16s_v2_aug import (
        MeniscusDatasetMT16SV2,
        load_mt_split,
        verify_no_leakage_mt,
    )

    tag = args.tag or args.ckpt.stem.replace("_best", "")
    tta_pipeline = build_tta_pipeline(args.no_tta)
    tta_tag = "no_tta" if args.no_tta else f"tta_{len(tta_pipeline)}x"
    out_json = cfg.RESULTS / f"{tag}_{tta_tag}_results.json"
    out_csv = cfg.RESULTS / f"{tag}_{tta_tag}_predictions.csv"
    existing = [p for p in (out_json, out_csv)
                if p.exists() and (p == out_json or args.save_predictions)]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing TTA outputs. Use a new --tag or "
            "--overwrite.\n" + "\n".join(str(p) for p in existing))

    print(f"Loading checkpoint: {args.ckpt}")
    model, ckpt, backbone, local_weights = load_model_from_checkpoint(
        args.ckpt, device)
    val_auc = ckpt.get("val_metrics", {}).get("auc")
    if val_auc is None:
        print(f"Backbone: {backbone} | best val AUC unavailable")
    else:
        print(f"Backbone: {backbone} | best val AUC {val_auc:.4f}")
    print(f"Local weights: {local_weights}")

    verify_no_leakage_mt()
    val_rows = load_mt_split("val")
    test_rows = load_mt_split("test")
    val_ds = MeniscusDatasetMT16SV2(val_rows, augment=False)
    test_ds = MeniscusDatasetMT16SV2(test_rows, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)} | BS: {args.bs}")
    print(f"TTA: {[name for name, _ in tta_pipeline]}")

    print("\nVALIDATION SET")
    val_prob, val_y, val_sev_prob, val_sev_y = predict_with_tta(
        model, val_loader, device, tta_pipeline,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        max_batches=args.max_val_batches)
    val_05 = binary_metrics_from_probs(val_prob, val_y, threshold=0.5)
    val_sev = severity_metrics_from_probs(val_sev_prob, val_sev_y)
    best_thr, best_metric, sweep = find_best_threshold(
        val_y, val_prob, metric=args.metric)
    val_cal = binary_metrics_from_probs(val_prob, val_y, threshold=best_thr)
    print(f"  AUC thr=0.5     : {val_05['auc']:.4f}")
    print(f"  F1 thr=0.5      : {val_05['f1']:.4f}")
    print(f"  Best threshold  : {best_thr:.4f} ({args.metric}={best_metric:.4f})")
    print(f"  F1 calibrated   : {val_cal['f1']:.4f}")

    print("\nTEST SET")
    torch.cuda.empty_cache()
    test_prob, test_y, test_sev_prob, test_sev_y = predict_with_tta(
        model, test_loader, device, tta_pipeline,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        max_batches=args.max_test_batches)
    test_05 = binary_metrics_from_probs(test_prob, test_y, threshold=0.5)
    test_cal = binary_metrics_from_probs(test_prob, test_y, threshold=best_thr)
    test_sev = severity_metrics_from_probs(test_sev_prob, test_sev_y)

    print("  --- Default threshold (0.5) ---")
    print(f"  AUC          : {test_05['auc']:.4f}")
    print(f"  PR-AUC       : {test_05['pr_auc']:.4f}")
    print(f"  F1           : {test_05['f1']:.4f}")
    print(f"  Sensitivity  : {test_05['sensitivity']:.3f}")
    print(f"  Specificity  : {test_05['specificity']:.3f}")
    print(f"  Severity Acc : {test_sev['sev_acc']:.4f}")
    print(f"\n  --- Calibrated threshold ({best_thr:.4f}) ---")
    print(f"  F1           : {test_cal['f1']:.4f}")
    print(f"  Bal Acc      : {test_cal['bal_acc']:.4f}")
    print(f"  Sensitivity  : {test_cal['sensitivity']:.3f}")
    print(f"  Specificity  : {test_cal['specificity']:.3f}")

    results = {
        "model": str(args.ckpt),
        "tag": tag,
        "backbone": backbone,
        "local_weights": str(local_weights),
        "tta_enabled": not args.no_tta,
        "n_tta_transforms": len(tta_pipeline),
        "tta_names": [name for name, _ in tta_pipeline],
        "threshold_metric": args.metric,
        "best_threshold": float(best_thr),
        "val": {
            "binary_threshold_0.5": val_05,
            "binary_threshold_calibrated": val_cal,
            "severity": val_sev,
        },
        "test": {
            "binary_threshold_0.5": test_05,
            "binary_threshold_calibrated": test_cal,
            "severity": test_sev,
        },
        "threshold_sweep_val": sweep[::10],
    }
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved -> {out_json}")

    if args.save_predictions:
        pd.DataFrame({
            "ID": test_rows["ID"].values,
            "SIDE": test_rows["SIDE"].values,
            "label": test_y.astype(int),
            "prob": test_prob,
            "pred_0p5": (test_prob >= 0.5).astype(int),
            "pred_calibrated": (test_prob >= best_thr).astype(int),
        }).to_csv(out_csv, index=False)
        print(f"Predictions saved -> {out_csv}")


if __name__ == "__main__":
    main()
