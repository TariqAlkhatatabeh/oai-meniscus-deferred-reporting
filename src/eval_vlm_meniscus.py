"""Zero-shot Qwen2.5-VL evaluation on the locked test set."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PIL import Image

import config as cfg

RESULTS = cfg.RESULTS
IMAGE_ROOT = cfg.DATA / "processed_16s_v2"
DEFAULT_MODEL = cfg.MODEL_QWEN_VL
TEST_CSV = RESULTS / "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"
OUT_DIR = RESULTS / "vlm_eval"

PROMPT = (
    "You are an expert musculoskeletal radiologist. These are knee MRI "
    "images (proton-density-style slices; sagittal DESS and/or reformatted "
    "planes). Task: estimate the probability that this knee has a "
    "meniscus tear. A tear means any meniscal region graded 2 or higher on "
    "the MOAKS scale (grade 1 signal abnormality alone is NOT a tear). "
    "Grade 2 or higher means linear high signal definitely reaching the "
    "articular surface, a visible tear line, or a missing meniscal piece. "
    "Respond with JSON only, no other text: "
    '{"p_tear": <float between 0 and 1>}'
)
RETRY_SUFFIX = " Respond with JSON only: {\"p_tear\": <float>}."


def parse_args():
    parser = argparse.ArgumentParser(description="Arm C VLM evaluation")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test_csv", type=Path, default=TEST_CSV)
    parser.add_argument("--tag", type=str, default="qwen25vl7b_zeroshot")
    parser.add_argument("--variants", type=str, default="sag_single,tri_grid,sag_montage")
    parser.add_argument("--limit", type=int, default=0,
                        help="debug: only first N knees")
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    return parser.parse_args()


def build_plane_montages(folder: Path) -> list:
    """Three 4x4 plane montages (sagittal DESS, coronal, axial) = all 48 images."""
    def load(prefix, index):
        return Image.open(folder / f"{prefix}_{index:02d}.png").convert("RGB")
    grids = []
    for prefix in ("dess", "cor_mpr", "ax_mpr"):
        tiles = [load(prefix, i) for i in range(16)]
        size = tiles[0].size
        gap = 4
        cols, rows = 4, 4
        canvas = Image.new(
            "RGB",
            (cols * size[0] + (cols - 1) * gap,
             rows * size[1] + (rows - 1) * gap),
            color=(255, 255, 255))
        for idx, tile in enumerate(tiles):
            col, row = idx % cols, idx // cols
            canvas.paste(tile, (col * (size[0] + gap), row * (size[1] + gap)))
        grids.append(canvas)
    return grids


def build_images(folder: Path, variant: str) -> Image.Image | None:
    def load(prefix: str, index: int) -> Image.Image:
        path = folder / f"{prefix}_{index:02d}.png"
        return Image.open(path).convert("RGB")

    if variant == "sag_single":
        return load("dess", 7)
    if variant == "sag_montage":
        tiles = [load("dess", i) for i in range(16)]
        cols, rows = 4, 4
    elif variant == "sag7_grid":
        tiles = [load("dess", i) for i in (0, 3, 5, 8, 10, 13, 15)]
        cols, rows = 4, 2
    elif variant == "tri_grid":
        tiles = [load(p, i) for p in ("dess", "cor_mpr", "ax_mpr")
                 for i in (3, 8, 12)]
        cols, rows = 3, 3
    else:
        raise ValueError(f"unknown variant: {variant}")
    size = tiles[0].size
    gap = 4
    canvas = Image.new(
        "RGB",
        (cols * size[0] + (cols - 1) * gap,
         rows * size[1] + (rows - 1) * gap),
        color=(255, 255, 255))
    for idx, tile in enumerate(tiles):
        col, row = idx % cols, idx // cols
        canvas.paste(tile, (col * (size[0] + gap), row * (size[1] + gap)))
    return canvas


def parse_probability(text: str) -> tuple[float, bool]:
    match = re.search(r'"?p_tear"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
    if not match:
        return float("nan"), False
    value = float(match.group(1))
    if not 0.0 <= value <= 1.0:
        return float("nan"), False
    return value, True


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{args.tag}_predictions.csv"
    log_path = OUT_DIR / f"{args.tag}.log"
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    done = set()
    records = []
    if out_csv.exists():
        existing = pd.read_csv(out_csv, dtype={"ID": str})
        records = existing.to_dict("records")
        done = {(str(r["ID"]), int(r["SIDE"]), r["variant"])
                for r in records}
        print(f"Resuming: {len(done)} knee-variant results already present")

    test = pd.read_csv(args.test_csv, dtype={"ID": str})
    if args.limit > 0:
        test = test.head(args.limit)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading {args.model_dir} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), quantization_config=quant,
        device_map="cuda", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28)
    model.eval()
    print("Model loaded.", flush=True)

    log_file = open(log_path, "a", encoding="utf-8")

    def log(line: str):
        stamp = time.strftime("%H:%M:%S")
        text = f"[{stamp}] {line}"
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    total = len(test) * len(variants)
    completed = 0
    skipped = sum(1 for _, r in test.iterrows()
                  if (str(r["ID"]), int(r["SIDE"]), "x") and
                  any((str(r["ID"]), int(r["SIDE"]), v) in done
                      for v in variants))
    t_start = time.time()

    for knee_idx, row in test.iterrows():
        knee_id = str(row["ID"])
        side = int(row["SIDE"])
        label = int(row["label"])
        folder = IMAGE_ROOT / f"{knee_id}_{side}"
        if not folder.exists():
            log(f"MISSING folder {folder}; skipping knee")
            continue
        for variant in variants:
            if (knee_id, side, variant) in done:
                continue
            if variant == "planes3_montage":
                images = build_plane_montages(folder)
                image = images[0]
                image_args = images
            else:
                image = build_images(folder, variant)
                images = [image]
                image_args = [image]
            attempt_texts = []
            prob, parse_ok = float("nan"), False
            for attempt in range(2):
                prompt = PROMPT if attempt == 0 else PROMPT + RETRY_SUFFIX
                content = [{"type": "image", "image": im}
                           for im in images] + \
                    [{"type": "text", "text": prompt}]
                messages = [{"role": "user", "content": content}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text], images=image_args, return_tensors="pt",
                ).to("cuda")
                with torch.no_grad():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False)
                trimmed = generated[:, inputs["input_ids"].shape[1]:]
                response = processor.batch_decode(
                    trimmed, skip_special_tokens=True)[0].strip()
                attempt_texts.append(response)
                prob, parse_ok = parse_probability(response)
                if parse_ok:
                    break
            records.append({
                "ID": knee_id, "SIDE": side, "label": label,
                "variant": variant,
                "p_tear": prob, "parse_ok": int(parse_ok),
                "attempts": len(attempt_texts),
                "response": " | ".join(attempt_texts)[:200],
            })
            completed += 1
            if completed % 10 == 0:
                rate = completed / (time.time() - t_start)
                remaining = (total - completed - skipped) / max(rate, 1e-9)
                log(f"{completed}/{total} knees-variants done, "
                    f"{rate:.2f}/s, ETA {remaining / 60:.0f} min")
            if completed % args.save_every == 0:
                pd.DataFrame(records).to_csv(out_csv, index=False)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    parse_fail = sum(1 for r in records if not r["parse_ok"])
    log(f"DONE: {len(records)} results saved to {out_csv}; "
        f"parse failures: {parse_fail}")
    log_file.close()


if __name__ == "__main__":
    main()
