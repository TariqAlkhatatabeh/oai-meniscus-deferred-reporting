"""Train the primary tri-plane ConvNeXt."""
import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import config as cfg


N_SLICES = 16
PROCESSED_INDEX = cfg.DATA / "processed_index_16s_v2.csv"
CONVNEXT_BASE_WEIGHTS = cfg.PRETRAINED_CONVNEXT
VIT_BASE_AUGREG_WEIGHTS = cfg.PRETRAINED_VIT
AUG_RECIPES = ["baseline", "randaug_mri_lite", "safe_mild", "safe_strong"]
BACKBONE_DEFAULT_WEIGHTS = {
    "convnext_base.fb_in22k_ft_in1k": CONVNEXT_BASE_WEIGHTS,
    "vit_base_patch16_224.augreg_in21k": VIT_BASE_AUGREG_WEIGHTS,
}


def _amp_settings(mode: str):
    if mode == "off":
        return False, torch.float32, False
    if mode == "bf16":
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _autocast(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def _move_x_to_device(x, device):
    for key in x:
        x[key] = x[key].to(device, non_blocking=True)
    return x


def _gpu_memory_gb():
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 3),
        torch.cuda.max_memory_reserved() / (1024 ** 3),
    )


def metrics_from(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    labels_int = labels.astype(int)
    preds = (probs >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(labels_int, probs))
        if len(np.unique(labels_int)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels_int, probs)),
        "bal_acc": float(balanced_accuracy_score(labels_int, preds)),
        "f1": float(f1_score(labels_int, preds, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(labels_int, preds, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return out


def severity_metrics(logits, labels):
    preds = logits.argmax(axis=1)
    labels_int = labels.astype(int)
    return {
        "sev_acc": float(accuracy_score(labels_int, preds)),
        "sev_f1_macro": float(f1_score(
            labels_int, preds, average="macro", zero_division=0)),
    }


def compute_loss(main_out, aux_out, y, severity, main_criterion,
                 aux_criterion, aux_weight):
    main_out = main_out.squeeze(-1)
    main_loss = main_criterion(main_out.float(), y)
    aux_loss = aux_criterion(aux_out.float(), severity)
    loss = main_loss + aux_weight * aux_loss
    return loss, main_loss, aux_loss


def apply_mixup(x, y, severity, alpha: float):
    if alpha <= 0.0 or y.size(0) < 2:
        return x, y, severity, False
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(y.size(0), device=y.device)
    mixed_x = {
        key: lam * value + (1.0 - lam) * value[index]
        for key, value in x.items()
    }
    mixed_y = lam * y + (1.0 - lam) * y[index]
    severity_onehot = torch.nn.functional.one_hot(
        severity, num_classes=3).float()
    mixed_severity = (
        lam * severity_onehot + (1.0 - lam) * severity_onehot[index]
    )
    return mixed_x, mixed_y, mixed_severity, True


def soft_cross_entropy(logits, soft_targets, class_weight=None):
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=1)
    if class_weight is not None:
        soft_targets = soft_targets * class_weight.view(1, -1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def compute_loss_mixup(main_out, aux_out, y_soft, severity_soft,
                       main_criterion, aux_weight, severity_weight):
    main_out = main_out.squeeze(-1)
    main_loss = main_criterion(main_out.float(), y_soft)
    aux_loss = soft_cross_entropy(
        aux_out.float(), severity_soft, class_weight=severity_weight)
    loss = main_loss + aux_weight * aux_loss
    return loss, main_loss, aux_loss


@torch.no_grad()
def evaluate(model, loader, device, main_criterion, aux_criterion,
             aux_weight, amp_enabled=False, amp_dtype=torch.float16,
             max_batches: int = 0):
    model.eval()
    all_main_logits, all_main_labels = [], []
    all_aux_logits, all_aux_labels = [], []
    total_main_loss, total_aux_loss, n = 0.0, 0.0, 0

    for batch_idx, (x, y, severity) in enumerate(loader):
        y = y.to(device)
        severity = severity.to(device)
        x = _move_x_to_device(x, device)
        with _autocast(amp_enabled, amp_dtype):
            main_out, aux_out = model(x)
            _loss, main_loss, aux_loss = compute_loss(
                main_out, aux_out, y, severity,
                main_criterion, aux_criterion, aux_weight)

        batch_size = y.size(0)
        total_main_loss += main_loss.item() * batch_size
        total_aux_loss += aux_loss.item() * batch_size
        n += batch_size
        all_main_logits.append(main_out.squeeze(-1).float().cpu().numpy())
        all_main_labels.append(y.float().cpu().numpy())
        all_aux_logits.append(aux_out.float().cpu().numpy())
        all_aux_labels.append(severity.long().cpu().numpy())

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    main_logits = np.concatenate(all_main_logits)
    main_labels = np.concatenate(all_main_labels)
    aux_logits = np.concatenate(all_aux_logits)
    aux_labels = np.concatenate(all_aux_labels)
    out = metrics_from(main_logits, main_labels)
    out.update(severity_metrics(aux_logits, aux_labels))
    out["loss"] = total_main_loss / max(n, 1)
    out["aux_loss"] = total_aux_loss / max(n, 1)
    out["combined_loss"] = out["loss"] + aux_weight * out["aux_loss"]
    return out


def write_history(history, tag):
    import pandas as pd

    log_dir = cfg.RESULTS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{tag}_history.json").write_text(
        json.dumps(history, indent=2))
    pd.DataFrame(history).to_csv(log_dir / f"{tag}_history.csv", index=False)
    return log_dir


def plot_training_curves(history, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = cfg.PROJECT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes[0, 0].plot(epochs, [h["train_main_loss"] for h in history],
                    label="Train Main", linewidth=2)
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history],
                    label="Val Main", linewidth=2)
    axes[0, 0].set_title("Main Task Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    axes[0, 1].plot(epochs, [h["train_aux_loss"] for h in history],
                    label="Train Aux", linewidth=2)
    axes[0, 1].plot(epochs, [h["val_aux_loss"] for h in history],
                    label="Val Aux", linewidth=2)
    axes[0, 1].set_title("Severity Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(epochs, [h["val_auc"] for h in history],
                    label="AUC", linewidth=2)
    axes[1, 0].plot(epochs, [h["val_f1"] for h in history],
                    label="F1", linewidth=2)
    axes[1, 0].set_title("Binary Validation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    axes[1, 1].plot(epochs, [h["val_sev_acc"] for h in history],
                    label="Severity Acc", linewidth=2)
    axes[1, 1].set_title("Auxiliary Task")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = out_dir / f"{tag}_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimental 16-slice multitask timm-backbone training")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=sorted(BACKBONE_DEFAULT_WEIGHTS))
    parser.add_argument("--local_weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--aug_recipe", type=str, default="safe_strong",
                        choices=AUG_RECIPES)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"],
                        default="fp16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--tag", type=str,
                        default="exp_multitask_16s_v2_timm")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--skip_final_eval", action="store_true")
    return parser.parse_args()


def resolve_weights(args) -> Path:
    path = args.local_weights or BACKBONE_DEFAULT_WEIGHTS[args.backbone]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def guard_outputs(args) -> None:
    if args.overwrite:
        return
    outputs = [
        cfg.WEIGHTS / f"{args.tag}_best.pt",
        cfg.RESULTS / "logs" / f"{args.tag}_history.csv",
        cfg.RESULTS / "logs" / f"{args.tag}_history.json",
        cfg.RESULTS / "logs" / f"{args.tag}_protocol.json",
        cfg.PROJECT / "figures" / f"{args.tag}_curves.png",
        cfg.RESULTS / f"{args.tag}_test_results.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing timm experiment outputs. Use a new "
            "--tag or pass --overwrite.\n" + "\n".join(str(p) for p in existing)
        )


def json_safe_args(args) -> dict:
    safe = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def write_protocol(args, weights_path: Path, describe_aug_recipe):
    log_dir = cfg.RESULTS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{args.tag}_protocol.json"
    payload = {
        "tag": args.tag,
        "script": "src/train_multitask_16s_v2_timm_exp.py",
        "dataset": "src/dataset_multitask_16s_v2_aug.py",
        "model": "src/model_multitask_16s_v2_timm_exp.py",
        "processed_index": str(PROCESSED_INDEX),
        "n_slices": N_SLICES,
        "recipe": "16-slice v2 multitask safe_strong timm-backbone experiment",
        "backbone": args.backbone,
        "local_weights": str(weights_path),
        "augmentation_recipe": args.aug_recipe,
        "augmentation_description": describe_aug_recipe(args.aug_recipe),
        "mixup_enabled": bool(args.mixup_alpha > 0.0),
        "mixup_alpha": args.mixup_alpha if args.mixup_alpha > 0.0 else None,
        "initialization": "local safetensors backbone only; no experiment checkpoint",
        "args": json_safe_args(args),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def set_encoder_state(model, trainable: bool) -> None:
    model.set_encoder_trainable(trainable)
    state = "trainable" if trainable else "frozen"
    print(f"Encoder is now {state}.")


def current_lrs(optimizer):
    return {group.get("name", f"group{idx}"): group["lr"]
            for idx, group in enumerate(optimizer.param_groups)}


def main():
    args = parse_args()
    weights_path = resolve_weights(args)
    guard_outputs(args)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available.")
    device = "cuda"
    amp_enabled, amp_dtype, use_scaler = _amp_settings(args.amp)
    torch.backends.cudnn.benchmark = True

    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    from dataset_multitask_16s_v2_aug import (
        describe_aug_recipe,
        get_mt_loaders,
        verify_no_leakage_mt,
    )
    from model_multitask_16s_v2_timm_exp import TriPlaneMultiTask16SV2Timm

    print(f"Tag: {args.tag} | 16-slice v2 timm-backbone experiment")
    print(f"Backbone: {args.backbone}")
    print(f"Local weights: {weights_path}")
    print("Init: local safetensors backbone only; no experiment checkpoint")
    print(f"BS={args.bs} grad_accum={args.grad_accum} "
          f"eff_batch={args.bs * args.grad_accum}")
    print(f"encoder_lr={args.encoder_lr} head_lr={args.head_lr} | "
          f"aux_weight={args.aux_weight} | amp={args.amp}")
    print(f"Freeze encoder epochs: {args.freeze_encoder_epochs}")
    print(f"Gradient checkpointing: {'ON' if args.grad_checkpointing else 'OFF'}")
    print(f"Augmentation: {args.aug_recipe} | "
          f"{describe_aug_recipe(args.aug_recipe)}")
    print(f"Mixup: {'ON alpha={:.2f}'.format(args.mixup_alpha) if args.mixup_alpha > 0 else 'OFF'}")

    verify_no_leakage_mt()
    train_loader, val_loader, test_loader = get_mt_loaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        aug_recipe=args.aug_recipe,
    )
    protocol_path = write_protocol(args, weights_path, describe_aug_recipe)
    print(f"Protocol: {protocol_path}")

    model = TriPlaneMultiTask16SV2Timm(
        args.backbone,
        weights_path,
        dropout=args.dropout,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    y_train = train_loader.dataset.rows["label"].astype(int).to_numpy()
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32, device=device)
    main_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    sev_train = train_loader.dataset.rows["severity"].to_numpy()
    sev_counts = np.bincount(sev_train, minlength=3).astype(float)
    sev_weight = sev_counts.sum() / (3.0 * np.maximum(sev_counts, 1.0))
    sev_weight = torch.tensor(sev_weight, dtype=torch.float32, device=device)
    aux_criterion = nn.CrossEntropyLoss(weight=sev_weight)

    print(f"Main pos_weight: {pos_weight.item():.3f}")
    print(f"Severity class weights: {sev_weight.cpu().numpy()}")

    optimizer = AdamW(
        [
            {"params": model.encoder_parameters(),
             "lr": args.encoder_lr, "name": "encoder"},
            {"params": model.head_parameters(),
             "lr": args.head_lr, "name": "head"},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )
    warmup = LinearLR(optimizer, start_factor=0.1,
                      total_iters=max(args.warmup, 1))
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup, 1))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup])
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_clip = getattr(cfg, "GRAD_CLIP", 1.0)

    best_auc = -1.0
    no_improve = 0
    history = []
    weight_path = cfg.WEIGHTS / f"{args.tag}_best.pt"
    cfg.WEIGHTS.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.RESULTS / "logs"

    print(f"\n{'=' * 70}")
    print(f"Starting timm-backbone training: {args.tag}")
    print(f"{'=' * 70}\n")

    encoder_trainable = args.freeze_encoder_epochs <= 0
    set_encoder_state(model, encoder_trainable)

    for epoch in range(1, args.epochs + 1):
        if (not encoder_trainable) and epoch > args.freeze_encoder_epochs:
            encoder_trainable = True
            set_encoder_state(model, True)

        torch.cuda.reset_peak_memory_stats()
        model.train()
        running_main, running_aux, n = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        optimizer_stepped = False
        batches_seen = 0

        phase = "unfrozen" if encoder_trainable else "frozen"
        pbar = tqdm(train_loader, desc=f"Ep {epoch:2d}/{args.epochs} {phase}",
                    leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
        for batch_idx, (x, y, severity) in enumerate(pbar):
            batches_seen = batch_idx + 1
            y = y.to(device)
            severity = severity.to(device)
            x = _move_x_to_device(x, device)
            if args.mixup_alpha > 0.0:
                x, train_y, train_severity, mixed = apply_mixup(
                    x, y, severity, args.mixup_alpha)
            else:
                train_y, train_severity, mixed = y, severity, False

            with _autocast(amp_enabled, amp_dtype):
                main_out, aux_out = model(x)
                if mixed:
                    loss, main_loss, aux_loss = compute_loss_mixup(
                        main_out, aux_out, train_y, train_severity,
                        main_criterion, args.aux_weight, sev_weight)
                else:
                    loss, main_loss, aux_loss = compute_loss(
                        main_out, aux_out, train_y, train_severity,
                        main_criterion, aux_criterion, args.aux_weight)

            scaler.scale(loss / args.grad_accum).backward()
            if batches_seen % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_stepped = optimizer_stepped or (
                        scaler.get_scale() >= old_scale)
                else:
                    optimizer.step()
                    optimizer_stepped = True
                optimizer.zero_grad(set_to_none=True)

            batch_size = y.size(0)
            n += batch_size
            running_main += main_loss.item() * batch_size
            running_aux += aux_loss.item() * batch_size
            pbar.set_postfix(
                main=f"{running_main / max(n, 1):.4f}",
                aux=f"{running_aux / max(n, 1):.4f}",
                comb=f"{(running_main + args.aux_weight * running_aux) / max(n, 1):.4f}",
            )

            if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
                break

        if batches_seen and batches_seen % args.grad_accum != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = optimizer_stepped or (
                    scaler.get_scale() >= old_scale)
            else:
                optimizer.step()
                optimizer_stepped = True
            optimizer.zero_grad(set_to_none=True)

        if optimizer_stepped:
            scheduler.step()

        train_main = running_main / max(n, 1)
        train_aux = running_aux / max(n, 1)
        lrs = current_lrs(optimizer)
        val = evaluate(
            model, val_loader, device, main_criterion, aux_criterion,
            args.aux_weight, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            max_batches=args.max_eval_batches)
        peak_alloc, peak_reserved = _gpu_memory_gb()

        log = {
            "epoch": epoch,
            "phase": phase,
            "backbone": args.backbone,
            "local_weights": str(weights_path),
            "aug_recipe": args.aug_recipe,
            "mixup_alpha": args.mixup_alpha,
            "train_main_loss": train_main,
            "train_aux_loss": train_aux,
            "encoder_lr": lrs.get("encoder"),
            "head_lr": lrs.get("head"),
            "peak_vram_alloc_gb": peak_alloc,
            "peak_vram_reserved_gb": peak_reserved,
            **{f"val_{key}": value for key, value in val.items()},
        }
        history.append(log)
        log_dir = write_history(history, args.tag)

        print(f"Ep {epoch:3d}/{args.epochs} [{phase}] | "
              f"tr_main {train_main:.4f} tr_aux {train_aux:.4f} | "
              f"val_loss {val['loss']:.4f} val_aux {val['aux_loss']:.4f} | "
              f"AUC {val['auc']:.4f} | F1 {val['f1']:.4f} | "
              f"sens {val['sensitivity']:.3f} spec {val['specificity']:.3f} | "
              f"sev_acc {val['sev_acc']:.3f} | "
              f"lr enc/head {lrs.get('encoder'):.2e}/{lrs.get('head'):.2e} | "
              f"VRAM {peak_alloc:.1f}/{peak_reserved:.1f} GB")

        if np.isfinite(val["auc"]) and val["auc"] > best_auc:
            best_auc = val["auc"]
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_metrics": val,
                "backbone": args.backbone,
                "local_weights": str(weights_path),
                "aux_weight": args.aux_weight,
                "dropout": args.dropout,
                "aug_recipe": args.aug_recipe,
                "mixup_alpha": args.mixup_alpha,
                "args": json_safe_args(args),
                "n_slices": N_SLICES,
                "processed_index": str(PROCESSED_INDEX),
                "recipe": "16-slice v2 timm-backbone experiment",
            }, weight_path)
            print(f"   -> saved best (AUC {best_auc:.4f}) -> {weight_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val-AUC gain for {args.patience} epochs).")
                break

    if args.skip_final_eval:
        plot_training_curves(history, args.tag)
        print("Skipping final test evaluation.")
        print(f"Training log: {log_dir / f'{args.tag}_history.csv'}")
        return

    print(f"\n{'=' * 70}")
    print(f"Loading best weights from {weight_path}")
    print(f"{'=' * 70}\n")
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(
        model, test_loader, device, main_criterion, aux_criterion,
        args.aux_weight, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        max_batches=args.max_eval_batches)
    (cfg.RESULTS / f"{args.tag}_test_results.json").write_text(
        json.dumps({
            "test": test,
            "best_val_auc": best_auc,
            "backbone": args.backbone,
            "local_weights": str(weights_path),
            "augmentation_recipe": args.aug_recipe,
            "augmentation_description": describe_aug_recipe(args.aug_recipe),
            "mixup_alpha": args.mixup_alpha if args.mixup_alpha > 0.0 else None,
        }, indent=2))
    plot_training_curves(history, args.tag)

    print("=== TEST SET (Binary Tear Detection) ===")
    print(f"AUC          : {test['auc']:.4f}")
    print(f"PR-AUC       : {test['pr_auc']:.4f}")
    print(f"F1           : {test['f1']:.4f}")
    print(f"Sensitivity  : {test['sensitivity']:.3f}")
    print(f"Specificity  : {test['specificity']:.3f}")
    print("\n=== TEST SET (Auxiliary Task) ===")
    print(f"Severity Acc : {test['sev_acc']:.3f}")
    print(f"Severity F1  : {test['sev_f1_macro']:.3f}")
    print(f"\nBest val AUC : {best_auc:.4f}")
    print(f"Best weights : {weight_path}")
    print(f"Training log : {log_dir / f'{args.tag}_history.csv'}")


if __name__ == "__main__":
    main()
