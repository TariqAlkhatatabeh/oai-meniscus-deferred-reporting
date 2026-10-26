"""Summarize the zero-shot VLM evaluation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import config as cfg

RESULTS = cfg.RESULTS
PRED_CSV = RESULTS / "vlm_eval" / "qwen25vl7b_all_variants.csv"
OUT_MD = RESULTS / "vlm_eval" / "qwen25vl7b_zeroshot_analysis.md"
OUT_CSV = RESULTS / "vlm_eval" / "qwen25vl7b_zeroshot_analysis_metrics.csv"
CLAIMS_CSV = cfg.PROJECT / "manuscript" / "evaluation" / "claims_table.csv"

VARIANT_LABELS = {
    "sag_single": "Single mid-sagittal DESS slice",
    "tri_grid": "3x3 tri-plane grid",
    "sag_montage": "4x4 sagittal montage (16 slices)",
    "planes3_montage": "Three plane montages (all 48 slices)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze VLM results")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--update_claims", action="store_true")
    return parser.parse_args()


def cluster_bootstrap_auc(df: pd.DataFrame, prob_col: str, n_boot: int,
                          rng: np.random.Generator) -> tuple:
    y = df["label"].to_numpy()
    p = df[prob_col].to_numpy()
    point = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    ids = df["ID"].drop_duplicates().to_numpy()
    index_by_id = {pid: np.where(df["ID"].to_numpy() == pid)[0] for pid in ids}
    samples = []
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([index_by_id[pid] for pid in sampled])
        ys, ps = y[idx], p[idx]
        samples.append(float(roc_auc_score(ys, ps))
                       if len(np.unique(ys)) > 1 else np.nan)
    arr = np.asarray(samples, dtype=float)
    finite = arr[np.isfinite(arr)]
    return point, float(np.percentile(finite, 2.5)), \
        float(np.percentile(finite, 97.5))


def main():
    args = parse_args()
    df = pd.read_csv(PRED_CSV, dtype={"ID": str})
    rng = np.random.default_rng(args.seed)

    rows = []
    for variant in df["variant"].drop_duplicates():
        block = df[df["variant"] == variant].reset_index(drop=True)
        point, low, high = cluster_bootstrap_auc(
            block, "p_tear", args.n_boot, rng)
        pred = (block["p_tear"] >= args.threshold).astype(int)
        bal_acc = float(balanced_accuracy_score(block["label"], pred))
        mean_pos = float(block.loc[block["label"] == 1, "p_tear"].mean())
        mean_neg = float(block.loc[block["label"] == 0, "p_tear"].mean())
        rows.append({
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "n_knees": len(block),
            "auc": round(point, 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
            "balanced_accuracy_0p5": round(bal_acc, 4),
            "mean_p_tear_label1": round(mean_pos, 3),
            "mean_p_tear_label0": round(mean_neg, 3),
            "parse_failures": int((block["parse_ok"] == 0).sum()),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_CSV, index=False)

    lines = [
        "# Arm C: Qwen2.5-VL-7B zero-shot meniscus tear detection",
        "",
        f"- Locked OAI test set: 566 knees; threshold {args.threshold}",
        f"- Bootstrap: participant-clustered, {args.n_boot} replicates",
        "- Reference: supervised ConvNeXt test AUC 0.9090 "
        "[0.8808-0.9349]",
        "",
        "| Input variant | AUC | 95% CI | Bal. acc @0.5 | "
        "Mean p(tear|label=1) | Mean p(tear|label=0) | Parse fails |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['auc']:.4f} | "
            f"{row['ci_low']:.4f}-{row['ci_high']:.4f} | "
            f"{row['balanced_accuracy_0p5']:.3f} | "
            f"{row['mean_p_tear_label1']:.3f} | "
            f"{row['mean_p_tear_label0']:.3f} | "
            f"{row['parse_failures']} |")
    OUT_MD.write_text("\n".join(lines) + "\n")

    if args.update_claims and CLAIMS_CSV.exists():
        claims = pd.read_csv(CLAIMS_CSV)
        for row in rows:
            key = f"vlm_{row['variant']}_auc"
            claims = claims[claims["claim_id"] != key]
            claims = pd.concat([claims, pd.DataFrame([{
                "claim_id": key,
                "metric": f"VLM zero-shot AUC ({row['label']})",
                "value": f"{row['auc']:.4f}",
                "ci95": f"{row['ci_low']:.4f}-{row['ci_high']:.4f}",
                "source_artifact": str(OUT_CSV),
            }])], ignore_index=True)
        claims.to_csv(CLAIMS_CSV, index=False)

    print(metrics.to_string(index=False))
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
