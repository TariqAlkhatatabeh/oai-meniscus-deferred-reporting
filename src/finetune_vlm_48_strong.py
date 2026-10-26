"""QLoRA adaptation of Qwen2.5-VL on the 48-image input."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

import config as cfg

MODEL_DIR = cfg.MODEL_QWEN_VL
IMAGE_ROOT = cfg.DATA / "processed_16s_v2"
SPLITS_CSV = cfg.DATA / "splits.csv"
LABELS_CSV = cfg.DATA / "labels.csv"
DEFAULT_OUT = cfg.RESULTS / "vlm_finetune_48_strong_v1"
ANSWER_PREFIX = '{"p_tear": '
PLANE_SPECS = (
    ("Sagittal DESS slices 01-16, ordered anterior to posterior.", "dess"),
    ("Coronal reformats 01-16, ordered consistently through the knee.", "cor_mpr"),
    ("Axial reformats 01-16, ordered superior to inferior.", "ax_mpr"),
)
PROMPT = (
    "Assess whether this knee has a meniscal tear using all ordered MRI slices. "
    "Return the probability of tear. Output JSON only."
)


def import_peft_symbols():
    """Import PEFT while bypassing bitsandbytes' irrelevant Windows Gaudi probe.

    bitsandbytes 0.49.2 calls the Linux-only shell pipeline
    ``pip list | grep habana-torch-plugin`` at import time. On this Windows host
    that subprocess can hang indefinitely. Returning an empty result only for
    this Gaudi query is correct on an NVIDIA-only workstation and avoids any
    environment or package modification.
    """
    original_run = subprocess.run

    def safe_run(command, *args, **kwargs):
        if isinstance(command, str) and "habana-torch-plugin" in command:
            return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="")
        return original_run(command, *args, **kwargs)

    subprocess.run = safe_run
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    finally:
        subprocess.run = original_run
    return LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "score", "smoke"), default="train")
    parser.add_argument("--model_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--resume_dir", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--splits_csv", type=Path, default=SPLITS_CSV)
    parser.add_argument("--labels_csv", type=Path, default=LABELS_CSV)
    parser.add_argument("--score_split", choices=("val", "test"), default="test")
    parser.add_argument("--score_out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.10)
    parser.add_argument("--input_mode", choices=("individual", "submontage2x2"),
                        default="submontage2x2")
    parser.add_argument("--image_size", type=int, default=336,
                        help="Processor size per input image; must be divisible by 28")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_vram_fraction", type=float, default=0.88,
                        help="Hard per-process CUDA allocator cap to protect WDDM")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args()


def load_knees(image_root: Path, splits_csv: Path, labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv, dtype={"ID": str})
    splits = pd.read_csv(splits_csv, dtype={"ID": str})
    knees = labels.merge(splits, on="ID", how="inner")
    knees = knees[knees["label"].isin([0, 1])].copy()
    knees["folder"] = [str(image_root / f"{pid}_{int(side)}")
                       for pid, side in zip(knees["ID"], knees["SIDE"])]
    expected_names = {f"{prefix}_{idx:02d}.png"
                      for _, prefix in PLANE_SPECS for idx in range(16)}

    def complete_folder(value):
        folder = Path(value)
        if not folder.is_dir():
            return False
        # One directory enumeration is substantially safer on the project drive
        # than 48 independent metadata calls for every knee.
        present = {item.name for item in folder.iterdir() if item.suffix.lower() == ".png"}
        return expected_names.issubset(present)

    knees["complete"] = knees["folder"].map(complete_folder)
    missing = knees[~knees["complete"]]
    if len(missing):
        print("Explicitly excluding image-incomplete knees:")
        print(missing[["ID", "SIDE", "label", "split"]].to_string(index=False))
    knees = knees[knees["complete"]].drop(columns="complete")
    counts = knees.groupby("split").size().to_dict()
    expected = {"train": 2566, "val": 540, "test": 566}
    if counts != expected:
        raise ValueError(f"Unexpected image-complete split counts: {counts}")
    if knees.duplicated(["ID", "SIDE"]).any():
        raise ValueError("Duplicate knee IDs detected")
    return knees.reset_index(drop=True)


def _augment_image(image: Image.Image, params: dict, noise_seed: int) -> Image.Image:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    image = TF.affine(
        image, angle=params["angle"], translate=params["translate"],
        scale=params["scale"], shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=0)
    image = ImageEnhance.Contrast(image).enhance(params["contrast"])
    image = ImageEnhance.Brightness(image).enhance(params["brightness"])
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(noise_seed).normal(0.0, params["noise"], array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="L")


def load_ordered_images(folder: Path, augment: bool, seed: int):
    rng = np.random.default_rng(seed)
    params = {
        "angle": float(rng.uniform(-5.0, 5.0)),
        "translate": (int(rng.integers(-8, 9)), int(rng.integers(-8, 9))),
        "scale": float(rng.uniform(0.95, 1.05)),
        "contrast": float(rng.uniform(0.90, 1.10)),
        "brightness": float(rng.uniform(0.92, 1.08)),
        "noise": float(rng.uniform(0.0, 2.5)),
    }
    images = []
    for plane_index, (_, prefix) in enumerate(PLANE_SPECS):
        for index in range(16):
            image = Image.open(folder / f"{prefix}_{index:02d}.png").convert("L")
            if augment:
                image = _augment_image(image, params, seed + plane_index * 100 + index)
            images.append(image.convert("RGB"))
    return images


def make_2x2_submontages(slices):
    montages = []
    gap = 4
    for start in range(0, len(slices), 4):
        block = slices[start:start + 4]
        width, height = block[0].size
        canvas = Image.new("RGB", (2 * width + gap, 2 * height + gap), color=(0, 0, 0))
        for index, image in enumerate(block):
            canvas.paste(image, ((index % 2) * (width + gap),
                                 (index // 2) * (height + gap)))
        montages.append(canvas)
    return montages


def build_messages(folder: Path, input_mode: str, augment: bool = False, seed: int = 0):
    slices = load_ordered_images(folder, augment=augment, seed=seed)
    content = []
    cursor = 0
    for description, _ in PLANE_SPECS:
        content.append({"type": "text", "text": description})
        plane_slices = slices[cursor:cursor + 16]
        images = (plane_slices if input_mode == "individual"
                  else make_2x2_submontages(plane_slices))
        for image in images:
            content.append({"type": "image", "image": image})
        cursor += 16
    content.append({"type": "text", "text": PROMPT})
    processor_images = [item["image"] for item in content if item["type"] == "image"]
    return processor_images, [{"role": "user", "content": content}]


def load_base(model_dir: Path, image_size: int):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration

    if image_size % 28:
        raise ValueError("--image_size must be divisible by 28")
    pixels = (image_size // 28) ** 2 * 28 * 28
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), quantization_config=quant, device_map="cuda",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True)
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(
        str(model_dir), min_pixels=pixels, max_pixels=pixels,
        local_files_only=True)
    return model, processor


def digit_ids(tokenizer):
    values = []
    for digit in ("1", "0"):
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Digit {digit} is not a single token: {encoded}")
        values.append(encoded[0])
    return values


def encode_case(processor, row, input_mode: str, augment: bool, seed: int):
    import torch

    images, messages = build_messages(
        Path(row["folder"]), input_mode=input_mode, augment=augment, seed=seed)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(
        text=[prompt + ANSWER_PREFIX], images=images, return_tensors="pt")
    return {key: value.to("cuda", non_blocking=True) if torch.is_tensor(value) else value
            for key, value in encoded.items()}


def class_logits(model, encoded, token_pair):
    output = model(**encoded)
    return output.logits[:, -1, :].index_select(-1, token_pair)


def classification_metrics(frame: pd.DataFrame, threshold=None) -> dict:
    y = frame["label"].to_numpy(int)
    p = frame["score"].to_numpy(float)
    if threshold is None:
        candidates = np.unique(np.r_[0.5, p])
        scored = [(f1_score(y, p >= value, zero_division=0), float(value))
                  for value in candidates]
        best = max(scored, key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))
    else:
        best = (f1_score(y, p >= float(threshold), zero_division=0), float(threshold))
    pred = p >= best[1]
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(best[0]), "accuracy": float(accuracy_score(y, pred)),
        "threshold": float(best[1]), "n": int(len(y)), "n_pos": int(y.sum()),
    }


def score_cases(model, processor, knees, token_pair, input_mode, desc):
    import torch

    was_training = model.training
    model.eval()
    records = []
    with torch.inference_mode():
        for _, row in tqdm(knees.iterrows(), total=len(knees), desc=desc):
            encoded = encode_case(
                processor, row, input_mode=input_mode, augment=False, seed=0)
            logits = class_logits(model, encoded, token_pair)
            probability = torch.softmax(logits.float(), dim=-1)[0, 0].item()
            records.append({"ID": str(row["ID"]), "SIDE": int(row["SIDE"]),
                            "label": int(row["label"]), "score": probability})
            del encoded, logits
    if was_training:
        model.train()
    return pd.DataFrame(records)


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    import torch

    warmup = max(1, round(total_steps * warmup_ratio))

    def scale(step):
        if step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale), warmup


def save_training_state(path, model, optimizer, scheduler, epoch, global_step,
                        best_auc, best_epoch, no_improve, history):
    import torch

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(item) for item in value)
        return value

    state = {
        "epoch": epoch, "global_step": global_step, "best_auc": best_auc,
        "best_epoch": best_epoch, "no_improve": no_improve, "history": history,
        "optimizer": cpu_copy(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    temporary = path / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(path / "training_state.pt")


def run_train(args, smoke: bool = False):
    print("Initializing strong-VLM training runtime...", flush=True)
    import torch
    import torch.nn.functional as F
    LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training = \
        import_peft_symbols()

    print("PyTorch/PEFT imports complete; initializing capped CUDA context...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not 0.50 <= args.max_vram_fraction <= 0.95:
        raise ValueError("--max_vram_fraction must be between 0.50 and 0.95")
    torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction, device=0)
    print(f"CUDA allocator cap set to {args.max_vram_fraction:.0%}.", flush=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.md"
    if log_path.exists() and not args.resume_dir:
        raise SystemExit(f"Refusing to append to existing run: {log_path}")

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    knees = load_knees(args.image_root, args.splits_csv, args.labels_csv)
    print("Strict 48-slice completeness audit passed.", flush=True)
    train = knees[knees["split"] == "train"].reset_index(drop=True)
    val = knees[knees["split"] == "val"].reset_index(drop=True)
    if args.train_limit:
        train = train.head(args.train_limit)
    if args.val_limit:
        val = val.head(args.val_limit)
    if smoke:
        train = pd.concat([train[train["label"] == 0].head(1),
                           train[train["label"] == 1].head(1)]).reset_index(drop=True)
        val = pd.concat([val[val["label"] == 0].head(1),
                         val[val["label"] == 1].head(1)]).reset_index(drop=True)
        args.max_steps = 1

    model, processor = load_base(args.model_dir, args.image_size)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.resume_dir:
        model = PeftModel.from_pretrained(
            model, str(args.resume_dir / "adapter"), is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules="all-linear",
            task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()
    torch.cuda.reset_peak_memory_stats()

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr,
            weight_decay=args.weight_decay)
        optimizer_name = "PagedAdamW8bit"
    except Exception as error:
        log(f"PagedAdamW8bit unavailable ({error}); using fused AdamW")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr,
            weight_decay=args.weight_decay, fused=True)
        optimizer_name = "AdamW(fused)"

    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler, warmup_steps = make_scheduler(
        optimizer, total_steps, args.warmup_ratio, args.min_lr_ratio)
    token_pair = torch.tensor(digit_ids(processor.tokenizer), device="cuda")
    counts = train["label"].value_counts().to_dict()
    if args.class_weight:
        n = len(train)
        class_weights = torch.tensor(
            [n / (2 * counts.get(1, 1)), n / (2 * counts.get(0, 1))],
            dtype=torch.float32, device="cuda")
    else:
        class_weights = None

    start_epoch = global_step = no_improve = 0
    best_auc, best_epoch, history = -1.0, 0, []
    if args.resume_dir:
        # This state is created locally by save_training_state and contains
        # optimizer plus Python/NumPy RNG objects, not only tensors.
        state = torch.load(
            args.resume_dir / "training_state.pt", map_location="cpu",
            weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step = int(state["epoch"]), int(state["global_step"])
        best_auc, best_epoch = float(state["best_auc"]), int(state["best_epoch"])
        no_improve, history = int(state["no_improve"]), state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        log(f"Resumed after epoch {start_epoch} at optimizer step {global_step}")

    config = vars(args).copy()
    input_description = ("48_ordered_individual_images" if args.input_mode == "individual"
                         else "12_ordered_2x2_submontages_from_48_source_images")
    config.update({"objective": "restricted_next_token_cross_entropy_1_vs_0",
                   "input": input_description, "optimizer": optimizer_name,
                   "warmup_steps": warmup_steps})
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in config.items()}
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Image-complete knees: train={len(train)}, val={len(val)}; "
        f"input={input_description}; processor image size={args.image_size}x{args.image_size}")
    log(f"Maximum epochs={args.epochs}; minimum epochs={args.min_epochs}; "
        f"early-stop patience={args.patience}; optimizer={optimizer_name}")

    run_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        order = train.sample(frac=1.0, random_state=args.seed + epoch).reset_index(drop=True)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_examples = 0
        progress = tqdm(order.iterrows(), total=len(order),
                        desc=f"Strong QLoRA epoch {epoch + 1}/{args.epochs}")
        for row_index, (_, row) in enumerate(progress):
            case_seed = args.seed * 1_000_003 + epoch * len(order) + row_index
            try:
                encoded = encode_case(
                    processor, row, input_mode=args.input_mode,
                    augment=args.augment, seed=case_seed)
                logits = class_logits(model, encoded, token_pair)
                target = torch.tensor([0 if int(row["label"]) == 1 else 1], device="cuda")
                raw_loss = F.cross_entropy(logits.float(), target, weight=class_weights)
                (raw_loss / args.grad_accum).backward()
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM at input_mode={args.input_mode}, image_size={args.image_size} "
                    f"under the {args.max_vram_fraction:.0%} safety cap. Reduce processor "
                    "image size; no training data were skipped.") from error
            epoch_loss += float(raw_loss.detach().cpu())
            n_examples += 1
            del encoded, logits, raw_loss, target
            if n_examples % args.grad_accum == 0 or n_examples == len(order):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0 or smoke:
                    progress.set_postfix(
                        loss=f"{epoch_loss / n_examples:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        grad=f"{float(grad_norm):.2f}")
                if args.max_steps and global_step >= args.max_steps:
                    break

        train_loss = epoch_loss / max(n_examples, 1)
        val_scores = score_cases(model, processor, val, token_pair, args.input_mode,
                                 f"validation epoch {epoch + 1}")
        val_scores.to_csv(args.out_dir / f"validation_scores_epoch{epoch + 1}.csv", index=False)
        val_metrics = classification_metrics(val_scores)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        epoch_record = {"epoch": epoch + 1, "train_loss": train_loss,
                        "lr": scheduler.get_last_lr()[0], "peak_gpu_gib": peak_gib,
                        **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
        log(f"Epoch {epoch + 1}: train_loss={train_loss:.5f}; "
            f"val_auc={val_metrics['auc']:.4f}; val_pr_auc={val_metrics['pr_auc']:.4f}; "
            f"val_f1={val_metrics['f1']:.4f}; peak_gpu={peak_gib:.2f} GiB")

        if val_metrics["auc"] > best_auc + 1e-5:
            best_auc, best_epoch, no_improve = val_metrics["auc"], epoch + 1, 0
            model.save_pretrained(args.out_dir / "lora_best")
            (args.out_dir / "best_info.json").write_text(json.dumps(
                {"epoch": best_epoch, "train_loss": train_loss,
                 "validation": val_metrics}, indent=2), encoding="utf-8")
            log(f"New best adapter: validation AUC {best_auc:.4f}")
        else:
            no_improve += 1

        del val_scores
        torch.cuda.empty_cache()
        save_training_state(args.out_dir / "resume_latest", model, optimizer,
                            scheduler, epoch + 1, global_step, best_auc,
                            best_epoch, no_improve, history)
        if smoke:
            log("Smoke completed successfully")
            return
        if args.max_steps and global_step >= args.max_steps:
            log("Stopped at requested max_steps")
            return
        if epoch + 1 >= args.min_epochs and no_improve >= args.patience:
            log(f"Early stopping at epoch {epoch + 1}; best epoch={best_epoch}, "
                f"best validation AUC={best_auc:.4f}")
            break

    elapsed_hours = (time.time() - run_start) / 3600
    log(f"Training complete in {elapsed_hours:.2f} h; best epoch={best_epoch}; "
        f"best validation AUC={best_auc:.4f}")


def run_score(args):
    import torch
    _, PeftModel, _, _ = import_peft_symbols()

    knees = load_knees(args.image_root, args.splits_csv, args.labels_csv)
    block = knees[knees["split"] == args.score_split].reset_index(drop=True)
    model, processor = load_base(args.model_dir, args.image_size)
    adapter = args.adapter_dir or (args.out_dir / "lora_best")
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    token_pair = torch.tensor(digit_ids(processor.tokenizer), device="cuda")
    scores = score_cases(model, processor, block, token_pair, args.input_mode,
                         f"score {args.score_split}")
    out = args.score_out or (args.out_dir / f"{args.score_split}_scores.csv")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    threshold = None
    if args.score_split == "test":
        info = json.loads((args.out_dir / "best_info.json").read_text(encoding="utf-8"))
        threshold = float(info["validation"]["threshold"])
    print(json.dumps(classification_metrics(scores, threshold), indent=2))


def main():
    args = parse_args()
    if args.mode == "score":
        run_score(args)
    else:
        run_train(args, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
