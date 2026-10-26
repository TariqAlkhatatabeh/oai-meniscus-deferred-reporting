"""Generate paired constrained/unconstrained reports for all 566 test knees.

Only outputs from the locked primary 16-slice ConvNeXt are exposed to the text
model. The script is resumable and writes to a new versioned results directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

import config as cfg

DEFAULT_MODEL = cfg.MODEL_QWEN_TEXT
TEST_CSV = PROJECT / "results" / "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"
CALIB_CSV = PROJECT / "results" / "convnext_safe_strong_16s_tta_20260629_calibration_test_predictions.csv"
SEVERITY_CSV = PROJECT / "results" / "reporting" / "primary16s_test_severity_tta7.csv"
COVERAGE_CSV = PROJECT / "results" / "metrics" / "convnext_safe_strong_16s_tta_20260629_coverage.csv"
OUT_DIR = PROJECT / "results" / "reporting"
DEFAULT_OUT = OUT_DIR / "qwen35_9b_primary16s_paired_reports.csv"
TARGET_COVERAGE = 0.8

SEVERITY_WORDS = {
    0: "grade 0 (no meniscal tear)",
    1: "grade 1 (intrasubstance signal without a surface-reaching tear)",
    2: "grade 2+ (surface-reaching meniscal tear)",
}

POLICY_PROMPT = """You are a radiology reporting assistant. Write a short knee MRI report in English with exactly two sections headed \"Findings:\" and \"Impression:\".

STRICT RULES:
1. Use only the facts in STRUCTURED FINDINGS. Do not add anatomical location, other structures, imaging features, measurements, or probabilities.
2. Do not provide treatment, follow-up, referral, or differential-diagnosis advice.
3. If WORKFLOW ACTION is AUTO_READ, make the Impression agree with the BINARY MODEL PREDICTION.
4. If WORKFLOW ACTION is DEFER, state that the automated assessment is inconclusive and radiologist review is required. Do not make a definitive tear or no-tear conclusion.
5. Present the severity output explicitly as an auxiliary model output, not as an independent clinical observation.
6. Use 3-6 sentences total.

STRUCTURED FINDINGS:
{facts}"""

UNCONSTRAINED_PROMPT = """You are a radiology reporting assistant. Write a short knee MRI report in English with sections headed \"Findings:\" and \"Impression:\" about the meniscus status of this knee.

STRUCTURED FINDINGS:
{facts}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=190)
    parser.add_argument("--limit-knees", type=int, default=0)
    return parser.parse_args()


def load_cutoff() -> float:
    coverage = pd.read_csv(COVERAGE_CSV)
    selected = coverage[
        coverage["split"].eq("val")
        & coverage["target_coverage"].astype(float).eq(TARGET_COVERAGE)
    ]
    if len(selected) != 1:
        raise ValueError("Could not uniquely resolve the validation-locked cutoff")
    return float(selected.iloc[0]["confidence_cutoff"])


def load_inputs() -> tuple[pd.DataFrame, float]:
    test = pd.read_csv(TEST_CSV, dtype={"ID": str})
    calibration = pd.read_csv(CALIB_CSV, dtype={"ID": str})
    severity = pd.read_csv(SEVERITY_CSV, dtype={"ID": str})
    merged = test.merge(
        calibration[["ID", "SIDE", "prob_raw", "prob_temperature_scaled"]],
        on=["ID", "SIDE"],
        validate="one_to_one",
    ).merge(
        severity[
            [
                "ID",
                "SIDE",
                "severity_pred",
                "severity_prob_grade0",
                "severity_prob_grade1",
                "severity_prob_grade2plus",
            ]
        ],
        on=["ID", "SIDE"],
        validate="one_to_one",
    )
    if len(merged) != len(test) or len(merged) != 566:
        raise ValueError(f"Expected 566 aligned test knees; found {len(merged)}")
    return merged, load_cutoff()


def build_facts(row: pd.Series, cutoff: float) -> dict:
    raw_prob = float(row["prob_raw"])
    confidence = max(raw_prob, 1.0 - raw_prob)
    action = "AUTO_READ" if confidence >= cutoff else "DEFER"
    severity_pred = int(row["severity_pred"])
    return {
        "knee": "right knee" if int(row["SIDE"]) == 1 else "left knee",
        "tta_tear_probability": round(raw_prob, 3),
        "temperature_scaled_tear_probability": round(
            float(row["prob_temperature_scaled"]), 3
        ),
        "binary_prediction": "tear" if int(row["pred_0p5"]) == 1 else "no tear",
        "model_confidence": round(confidence, 3),
        "workflow_action": action,
        "auxiliary_severity_prediction": SEVERITY_WORDS[severity_pred],
        "auxiliary_severity_probability": round(
            float(
                row[
                    [
                        "severity_prob_grade0",
                        "severity_prob_grade1",
                        "severity_prob_grade2plus",
                    ][severity_pred]
                ]
            ),
            3,
        ),
    }


def facts_text(facts: dict) -> str:
    return "\n".join(
        [
            f"- Knee: {facts['knee']}",
            f"- TTA tear probability: {facts['tta_tear_probability']:.3f}",
            "- Temperature-scaled tear probability: "
            f"{facts['temperature_scaled_tear_probability']:.3f}",
            f"- Binary model prediction: {facts['binary_prediction']}",
            f"- Model confidence: {facts['model_confidence']:.3f}",
            f"- Workflow action: {facts['workflow_action']}",
            "- Auxiliary severity-head prediction: "
            f"{facts['auxiliary_severity_prediction']}",
            "- Probability assigned to that severity class: "
            f"{facts['auxiliary_severity_probability']:.3f}",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")

    merged, cutoff = load_inputs()
    if args.limit_knees:
        merged = merged.head(args.limit_knees)
    print(f"Aligned knees: {len(merged)}; validation-locked cutoff: {cutoff:.4f}")

    records: list[dict] = []
    done: set[tuple[str, int, str]] = set()
    if args.out.exists():
        existing = pd.read_csv(args.out, dtype={"ID": str})
        records = existing.to_dict("records")
        done = {
            (str(record["ID"]), int(record["SIDE"]), str(record["arm"]))
            for record in records
        }
        print(f"Resuming with {len(done)} completed reports")

    tasks = []
    for arm in ("constrained", "unconstrained"):
        for _, row in merged.iterrows():
            key = (str(row["ID"]), int(row["SIDE"]), arm)
            if key in done:
                continue
            facts = build_facts(row, cutoff)
            prompt_template = POLICY_PROMPT if arm == "constrained" else UNCONSTRAINED_PROMPT
            tasks.append(
                {
                    "ID": key[0],
                    "SIDE": key[1],
                    "label": int(row["label"]),
                    "arm": arm,
                    "action": facts["workflow_action"],
                    "facts_json": json.dumps(facts, sort_keys=True),
                    "prompt": prompt_template.format(facts=facts_text(facts)),
                }
            )
    print(f"Reports remaining: {len(tasks)}")
    if not tasks:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        quantization_config=quantization,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print("Loaded Qwen3.5-9B in 4-bit NF4", flush=True)

    def write_log(message: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    completed = 0
    started = time.time()
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        conversations = [[{"role": "user", "content": item["prompt"]}] for item in batch]
        rendered = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in conversations
        ]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to("cuda")
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during report generation; rerun resumably with --batch-size 1"
            ) from exc
        prompt_width = inputs["input_ids"].shape[1]
        responses = tokenizer.batch_decode(
            output[:, prompt_width:], skip_special_tokens=True
        )
        for item, response in zip(batch, responses):
            record = {key: value for key, value in item.items() if key != "prompt"}
            record["report"] = response.strip()
            records.append(record)
            completed += 1
        if completed % args.save_every < len(batch) or start + len(batch) == len(tasks):
            pd.DataFrame(records).to_csv(args.out, index=False)
        if completed % 10 < len(batch) or start + len(batch) == len(tasks):
            rate = completed / max(time.time() - started, 1e-9)
            eta = (len(tasks) - completed) / max(rate, 1e-9) / 60
            write_log(f"{completed}/{len(tasks)} new reports; {rate:.3f}/s; ETA {eta:.1f} min")

    final = pd.DataFrame(records)
    final = final.sort_values(["arm", "ID", "SIDE"]).reset_index(drop=True)
    final.to_csv(args.out, index=False)
    write_log(f"DONE: {len(final)} total reports saved to {args.out}")


if __name__ == "__main__":
    main()
