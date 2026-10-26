"""Adapt the primary ConvNeXt to MRNet."""
from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        description="Domain adaptation of the primary 16-slice ConvNeXt on MRNet"
    )
    parser.add_argument(
        "--index", type=Path, default=cfg.DATA / "mrnet_primary_16s_v1_index.csv"
    )
    parser.add_argument(
        "--splits", type=Path, default=cfg.DATA / "mrnet_da_splits.csv"
    )
    parser.add_argument("--oai_ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tag", default="mrnet_primary_16s_da_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--minimum_epochs", type=int, default=10)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_batches", type=int, default=0)
    parser.add_argument("--resume_state", type=Path, default=None)
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def move_x(x: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in x.items()}


def autocast(enabled: bool, dtype: torch.dtype):
    return torch.amp.autocast("cuda", dtype=dtype) if enabled else nullcontext()


def metrics(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict:
    y = y.astype(int)
    if threshold is None:
        candidates = np.linspace(0.01, 0.99, 200)
        scores = [f1_score(y, p >= t, zero_division=0) for t in candidates]
        threshold = float(candidates[int(np.argmax(scores))])
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "loss": float(nn.functional.binary_cross_entropy(
            torch.from_numpy(p).float().clamp(1e-7, 1 - 1e-7),
            torch.from_numpy(y).float(),
        )),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
    }


def set_encoder_frozen(model: nn.Module, frozen: bool) -> None:
    for param in model.branches.parameters():
        param.requires_grad = not frozen
    for param in model.aux_head.parameters():
        param.requires_grad = False
    for param in model.main_head.parameters():
        param.requires_grad = True


@torch.no_grad()
def predict_single(model, loader, device, amp_enabled, amp_dtype, description):
    model.eval()
    probs, labels = [], []
    for x, y, _ in tqdm(loader, desc=description, leave=False):
        x = move_x(x, device)
        with autocast(amp_enabled, amp_dtype):
            logits, _ = model(x)
        probs.append(torch.sigmoid(logits.squeeze(-1).float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    for path in (args.index, args.splits, args.oai_ckpt):
        if not path.exists():
            raise FileNotFoundError(path)

    outputs = {
        "checkpoint": cfg.WEIGHTS / f"{args.tag}_best.pt",
        "state": cfg.RESULTS / "checkpoints" / f"{args.tag}_state.pt",
        "history": cfg.RESULTS / "logs" / f"{args.tag}_history.csv",
        "results": cfg.RESULTS / "metrics" / f"{args.tag}_results.json",
        "predictions": cfg.RESULTS / "metrics" / f"{args.tag}_test_predictions.csv",
        "curves": cfg.PROJECT / "figures" / f"{args.tag}_training_curves.png",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and args.resume_state is None:
        raise SystemExit(
            "Refusing to overwrite existing adaptation artifacts:\n"
            + "\n".join(str(path) for path in existing)
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    amp_enabled = args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    index = pd.read_csv(args.index, dtype={"ID": str})
    split_df = pd.read_csv(args.splits, dtype={"ID": str})[["ID", "da_split"]]
    rows = index.merge(split_df, on="ID", how="inner", validate="one_to_one")
    if len(rows) != len(index):
        raise ValueError("The fixed split file does not cover every MRNet examination")
    expected = [f"{seq}_{i:02d}" for seq in cfg.SEQUENCES for i in range(16)]
    if any(col not in rows.columns for col in expected):
        raise ValueError("MRNet index is not the required 16-slice tri-plane index")
    rows["severity"] = 0

    train_rows = rows[rows.da_split == "train"].copy()
    val_rows = rows[rows.da_split == "val"].copy()
    test_rows = rows[rows.da_split == "test"].copy()
    if [len(train_rows), len(val_rows), len(test_rows)] != [874, 188, 188]:
        raise ValueError("Unexpected fixed split sizes")

    train_ds = MeniscusDatasetMT16SV2(
        train_rows, augment=True, augment_recipe="safe_strong"
    )
    val_ds = MeniscusDatasetMT16SV2(val_rows, augment=False)
    test_ds = MeniscusDatasetMT16SV2(test_rows, augment=False)
    loader_kwargs = {
        "batch_size": args.bs,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model, source_ckpt, backbone, local_weights = load_model_from_checkpoint(
        args.oai_ckpt, device
    )
    for branch in model.branches.values():
        if hasattr(branch.net, "set_grad_checkpointing"):
            branch.net.set_grad_checkpointing(True)
    set_encoder_frozen(model, frozen=args.freeze_epochs > 0)

    n_pos = int(train_rows.label.sum())
    n_neg = len(train_rows) - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], device=device)
    )
    optimizer = AdamW(
        [
            {"params": list(model.branches.parameters()), "lr": args.encoder_lr},
            {"params": list(model.main_head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    if args.warmup_epochs:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=args.warmup_epochs,
                ),
                CosineAnnealingLR(
                    optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
                ),
            ],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Source checkpoint: {args.oai_ckpt}")
    print(f"Backbone: {backbone}; inputs: 16 slices x 3 planes")
    print(f"Split: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Effective batch: {args.bs * args.grad_accum}")

    best_auc = -np.inf
    best_epoch = 0
    best_threshold = 0.5
    no_improvement = 0
    history = []
    start_epoch = 1
    outputs["state"].parent.mkdir(parents=True, exist_ok=True)

    if args.resume_state is not None:
        if not args.resume_state.exists():
            raise FileNotFoundError(args.resume_state)
        resume = torch.load(args.resume_state, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        best_auc = float(resume["best_val_auc"])
        best_epoch = int(resume["best_epoch"])
        best_threshold = float(resume["best_val_threshold"])
        no_improvement = int(resume["epochs_without_improvement"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        print(f"Resumed complete training state after epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        set_encoder_frozen(model, frozen=epoch <= args.freeze_epochs)
        model.train()
        model.aux_head.eval()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, train_probs, train_labels = 0.0, [], []
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} train")
        for batch_idx, (x, y, _) in enumerate(pbar, start=1):
            x = move_x(x, device)
            y_dev = y.to(device, non_blocking=True)
            with autocast(amp_enabled, amp_dtype):
                logits, _ = model(x)
                raw_loss = criterion(logits.squeeze(-1), y_dev)
                loss = raw_loss / args.grad_accum
            scaler.scale(loss).backward()
            if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(raw_loss.detach())
            steps += 1
            train_probs.append(torch.sigmoid(logits.squeeze(-1).float()).detach().cpu().numpy())
            train_labels.append(y.numpy())
            pbar.set_postfix(loss=f"{epoch_loss / steps:.4f}")
            if args.smoke_batches and batch_idx >= args.smoke_batches:
                break

        train_p = np.concatenate(train_probs)
        train_y = np.concatenate(train_labels).astype(int)
        train_m = metrics(train_y, train_p)
        train_m["loss"] = epoch_loss / max(steps, 1)
        val_p, val_y = predict_single(
            model, val_loader, device, amp_enabled, amp_dtype, f"Epoch {epoch:02d} val"
        )
        val_m = metrics(val_y, val_p)
        scheduler.step()
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "train_f1": train_m["f1"],
            "train_roc_auc": train_m["roc_auc"],
            "train_pr_auc": train_m["pr_auc"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"],
            "val_pr_auc": val_m["pr_auc"],
            "val_threshold": val_m["threshold"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "peak_cuda_gib": peak_gib,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(outputs["history"], index=False)
        print(
            f"Epoch {epoch:02d}: train AUC {train_m['roc_auc']:.4f}, "
            f"val AUC {val_m['roc_auc']:.4f}, val F1 {val_m['f1']:.4f}, "
            f"peak {peak_gib:.2f} GiB"
        )

        if val_m["roc_auc"] > best_auc:
            best_auc = val_m["roc_auc"]
            best_epoch = epoch
            best_threshold = val_m["threshold"]
            no_improvement = 0
            checkpoint = {
                "model": model.state_dict(),
                "args": jsonable_args(args),
                "backbone": backbone,
                "local_weights": str(local_weights),
                "source_oai_checkpoint": str(args.oai_ckpt),
                "best_epoch": best_epoch,
                "best_val_auc": best_auc,
                "best_val_threshold_single_pass": best_threshold,
                "mrnet_split_file": str(args.splits),
                "n_slices": 16,
            }
            temp = outputs["checkpoint"].with_suffix(".tmp")
            outputs["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, temp)
            temp.replace(outputs["checkpoint"])
        else:
            no_improvement += 1

        state = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "best_val_threshold": best_threshold,
            "epochs_without_improvement": no_improvement,
            "history": history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        temp_state = outputs["state"].with_suffix(".tmp")
        torch.save(state, temp_state)
        temp_state.replace(outputs["state"])
        if args.smoke_batches:
            print("Smoke run completed; no held-out test evaluation performed")
            return
        if epoch >= args.minimum_epochs and no_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; selected epoch {best_epoch}")
            break

    model, selected_ckpt, _, _ = load_model_from_checkpoint(outputs["checkpoint"], device)
    tta = build_tta_pipeline(False)
    val_p, val_y, _, _ = predict_with_tta(
        model, val_loader, device, tta, amp_enabled=amp_enabled, amp_dtype=amp_dtype
    )
    val_tta = metrics(val_y.astype(int), val_p)
    test_p, test_y, _, _ = predict_with_tta(
        model, test_loader, device, tta, amp_enabled=amp_enabled, amp_dtype=amp_dtype
    )
    test_tta = metrics(test_y.astype(int), test_p, threshold=val_tta["threshold"])

    results = {
        "analysis": "domain adaptation of exact primary 16-slice ConvNeXt",
        "source_oai_checkpoint": str(args.oai_ckpt),
        "selected_checkpoint": str(outputs["checkpoint"]),
        "architecture_changed": False,
        "n_slices_per_plane": 16,
        "mrnet_fixed_split": str(args.splits),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "best_epoch": selected_ckpt["best_epoch"],
        "best_validation_auc_single_pass": selected_ckpt["best_val_auc"],
        "validation_tta_metrics": val_tta,
        "test_tta_metrics": test_tta,
        "recipe": jsonable_args(args),
    }
    outputs["results"].parent.mkdir(parents=True, exist_ok=True)
    outputs["results"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    predictions = test_rows[["ID", "split", "da_split", "label"]].copy()
    predictions["probability"] = test_p
    predictions.to_csv(outputs["predictions"], index=False)

    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(hist.epoch, hist.train_loss, label="Training")
    axes[0].plot(hist.epoch, hist.val_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Loss")
    axes[1].plot(hist.epoch, hist.train_roc_auc, label="Training")
    axes[1].plot(hist.epoch, hist.val_roc_auc, label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="ROC AUC", title="Discrimination")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    outputs["curves"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["curves"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
