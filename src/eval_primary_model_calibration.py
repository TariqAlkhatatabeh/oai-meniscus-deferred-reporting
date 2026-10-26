"""Temperature scaling and calibration metrics."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)

import config as cfg


DEFAULT_TAG = "convnext_safe_strong_16s_tta_20260629_calibration"
DEFAULT_VAL = cfg.RESULTS / "convnext_safe_strong_16s_tta_20260629_coverage_val_predictions.csv"
DEFAULT_TEST = cfg.RESULTS / "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration metrics and reliability figure")
    parser.add_argument("--val_csv", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test_csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def guard_outputs(paths, overwrite: bool):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing calibration outputs. Use a new "
            "--tag or pass --overwrite.\n" + "\n".join(str(p) for p in existing)
        )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "SIDE", "label", "prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["label"] = df["label"].astype(int)
    df["prob"] = df["prob"].astype(float).clip(1e-6, 1 - 1e-6)
    return df


def logit_np(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(prob / (1.0 - prob))


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_temperature(val_prob, val_y):
    logits = torch.tensor(logit_np(val_prob), dtype=torch.float32)
    labels = torch.tensor(np.asarray(val_y, dtype=float), dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).detach().clamp(0.05, 20.0))
    val_nll = float(F.binary_cross_entropy_with_logits(
        logits / temperature, labels).detach())
    return temperature, val_nll


def apply_temperature(prob, temperature):
    return sigmoid_np(logit_np(prob) / temperature)


def calibration_bins(y, prob, n_bins: int = 10):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        n = int(mask.sum())
        if n:
            mean_pred = float(prob[mask].mean())
            observed = float(y[mask].mean())
            gap = observed - mean_pred
            abs_gap = abs(gap)
            weight = n / len(y)
            ece += weight * abs_gap
            mce = max(mce, abs_gap)
        else:
            mean_pred = float("nan")
            observed = float("nan")
            gap = float("nan")
            abs_gap = float("nan")
        rows.append({
            "bin": idx + 1,
            "bin_left": float(left),
            "bin_right": float(right),
            "n": n,
            "mean_predicted_probability": mean_pred,
            "observed_positive_fraction": observed,
            "gap_observed_minus_predicted": gap,
            "abs_gap": abs_gap,
        })
    return rows, float(ece), float(mce)


def metrics_for(y, prob, threshold: float, n_bins: int):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float).clip(1e-6, 1 - 1e-6)
    pred = (prob >= threshold).astype(int)
    _bins, ece, mce = calibration_bins(y, prob, n_bins)
    return {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, prob)),
        "nll": float(log_loss(y, prob, labels=[0, 1])),
        "ece": ece,
        "mce": mce,
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, prob))
        if len(np.unique(y)) > 1 else float("nan"),
        "f1_at_0p5": float(f1_score(y, pred, zero_division=0)),
        "mean_probability": float(prob.mean()),
        "positive_rate": float(y.mean()),
    }


def build_bin_table(split_name, variant_name, y, prob, n_bins):
    rows, _ece, _mce = calibration_bins(y, prob, n_bins)
    for row in rows:
        row["split"] = split_name
        row["variant"] = variant_name
    return rows


def write_markdown(path, payload):
    test_raw = payload["metrics"]["test"]["raw"]
    test_temp = payload["metrics"]["test"]["temperature_scaled"]
    val_raw = payload["metrics"]["val"]["raw"]
    val_temp = payload["metrics"]["val"]["temperature_scaled"]
    lines = [
        "# Primary Model Calibration - 2026-06-28",
        "",
        "Locked model: `safe_strong 16-slice multitask ResNet50 mean-pooling + 7x TTA`.",
        "",
        f"Temperature was fit on validation only: `T = {payload['temperature']:.4f}`.",
        "",
        "## Metrics",
        "",
        "| Split | Variant | Brier | ECE | MCE | NLL | AUC | PR-AUC | F1 @0.5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, raw, temp in (
        ("Validation", val_raw, val_temp),
        ("Test", test_raw, test_temp),
    ):
        for variant_name, metrics in (("Raw", raw), ("Temperature scaled", temp)):
            lines.append(
                f"| {split_name} | {variant_name} | "
                f"{metrics['brier']:.4f} | {metrics['ece']:.4f} | "
                f"{metrics['mce']:.4f} | {metrics['nll']:.4f} | "
                f"{metrics['auc']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{metrics['f1_at_0p5']:.4f} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Temperature scaling was fit on validation probabilities only and then applied once to test.",
        "- AUC and F1 at threshold 0.5 are unchanged because scalar temperature scaling is monotonic and preserves the sign of the logit.",
        "- Calibration should be judged mainly by Brier score, ECE, MCE, NLL, and reliability curves.",
        "",
        "## Artifacts",
        "",
        f"- `{payload['artifacts']['json']}`",
        f"- `{payload['artifacts']['metrics_csv']}`",
        f"- `{payload['artifacts']['bins_csv']}`",
        f"- `{payload['artifacts']['figure']}`",
        f"- `{payload['artifacts']['figure_pdf']}`",
        f"- `{payload['artifacts']['caption']}`",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_caption(path, payload):
    test_raw = payload["metrics"]["test"]["raw"]
    test_temp = payload["metrics"]["test"]["temperature_scaled"]
    text = (
        "Figure. Calibration of the locked primary meniscus tear classifier. "
        "The model is the safe_strong 16-slice multitask ResNet50 mean-pooling "
        "network evaluated with 7x test-time augmentation. (A) Reliability "
        "curves on the held-out test set before and after validation-fitted "
        "temperature scaling; the diagonal indicates perfect calibration. "
        "(B) Per-bin calibration gap, computed as observed positive fraction "
        "minus mean predicted probability. (C) Distribution of predicted tear "
        "probabilities by true class after temperature scaling. (D) Calibration "
        "metrics comparing raw and temperature-scaled probabilities on test. "
        f"Temperature was fitted using validation data only (T={payload['temperature']:.3f}) "
        "and then applied once to the test set. On test, temperature scaling "
        f"changed Brier score from {test_raw['brier']:.3f} to {test_temp['brier']:.3f}, "
        f"ECE from {test_raw['ece']:.3f} to {test_temp['ece']:.3f}, and NLL "
        f"from {test_raw['nll']:.3f} to {test_temp['nll']:.3f}."
    )
    path.write_text(text + "\n")


def plot_calibration_figure(path, pdf_path, bins_df, metrics_df, test_df, n_bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
    })

    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
    grid = GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    colors = {"raw": "#2f6fbb", "temperature_scaled": "#d95f02"}
    labels = {"raw": "Raw", "temperature_scaled": "Temp scaled"}
    test_bins = bins_df[bins_df["split"] == "test"].copy()

    ax_a.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    for variant in ("raw", "temperature_scaled"):
        sub = test_bins[(test_bins["variant"] == variant) & (test_bins["n"] > 0)]
        ax_a.plot(
            sub["mean_predicted_probability"],
            sub["observed_positive_fraction"],
            marker="o", linewidth=1.5, markersize=4,
            color=colors[variant], label=labels[variant],
        )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Mean predicted probability")
    ax_a.set_ylabel("Observed tear fraction")
    ax_a.set_title("Reliability")
    ax_a.legend(frameon=False, loc="upper left")
    ax_a.grid(True, linestyle=":", alpha=0.45)

    width = 0.038
    centers = np.linspace(0.05, 0.95, n_bins)
    for offset, variant in ((-width / 2, "raw"), (width / 2, "temperature_scaled")):
        sub = test_bins[test_bins["variant"] == variant].sort_values("bin")
        gaps = sub["gap_observed_minus_predicted"].fillna(0.0).to_numpy()
        ax_b.bar(
            centers + offset, gaps, width=width,
            color=colors[variant], alpha=0.85, label=labels[variant],
        )
    ax_b.axhline(0, color="0.3", linewidth=0.8)
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("Predicted probability bin")
    ax_b.set_ylabel("Observed - predicted")
    ax_b.set_title("Calibration gap")
    ax_b.legend(frameon=False, loc="lower left")
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.45)

    bins = np.linspace(0, 1, 21)
    neg = test_df.loc[test_df["label"] == 0, "prob_temperature_scaled"]
    pos = test_df.loc[test_df["label"] == 1, "prob_temperature_scaled"]
    ax_c.hist(neg, bins=bins, density=False, alpha=0.72,
              color="#4c78a8", label="No tear")
    ax_c.hist(pos, bins=bins, density=False, alpha=0.72,
              color="#f58518", label="Tear")
    ax_c.axvline(0.5, color="0.25", linestyle="--", linewidth=1)
    ax_c.set_xlabel("Temperature-scaled probability")
    ax_c.set_ylabel("Number of knees")
    ax_c.set_title("Probability distribution", pad=18)
    ax_c.legend(frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.08), ncol=2)
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.45)

    metric_names = ["brier", "ece", "nll"]
    metric_labels = ["Brier", "ECE", "NLL"]
    test_metrics = metrics_df[metrics_df["split"] == "test"].set_index("variant")
    x = np.arange(len(metric_names))
    raw_values = [test_metrics.loc["raw", metric] for metric in metric_names]
    temp_values = [test_metrics.loc["temperature_scaled", metric] for metric in metric_names]
    ax_d.bar(x - 0.18, raw_values, width=0.36, color=colors["raw"], label="Raw")
    ax_d.bar(x + 0.18, temp_values, width=0.36,
             color=colors["temperature_scaled"], label="Temp scaled")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(metric_labels)
    ax_d.set_ylabel("Metric value")
    ax_d.set_title("Test calibration metrics")
    ax_d.legend(frameon=False, loc="upper left")
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.45)

    for label, ax in zip(("A", "B", "C", "D"), (ax_a, ax_b, ax_c, ax_d)):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figure_dir = cfg.PROJECT / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / f"{args.tag}.json"
    metrics_csv = metrics_dir / f"{args.tag}_metrics.csv"
    bins_csv = metrics_dir / f"{args.tag}_reliability_bins.csv"
    calibrated_pred_csv = cfg.RESULTS / f"{args.tag}_test_predictions.csv"
    out_md = logs_dir / f"{args.tag}.md"
    caption_path = logs_dir / f"{args.tag}_figure_caption.txt"
    figure_path = figure_dir / f"{args.tag}.png"
    figure_pdf_path = figure_dir / f"{args.tag}.pdf"
    guard_outputs(
        [out_json, metrics_csv, bins_csv, calibrated_pred_csv,
         out_md, caption_path, figure_path, figure_pdf_path],
        args.overwrite,
    )

    val_df = load_predictions(args.val_csv)
    test_df = load_predictions(args.test_csv)
    temperature, val_temp_nll = fit_temperature(
        val_df["prob"].to_numpy(), val_df["label"].to_numpy())
    print(f"Fitted validation temperature: {temperature:.4f}")
    print(f"Validation NLL after temperature scaling: {val_temp_nll:.4f}")

    for df in (val_df, test_df):
        df["prob_raw"] = df["prob"].astype(float)
        df["prob_temperature_scaled"] = apply_temperature(
            df["prob_raw"].to_numpy(), temperature)
        df["pred_temperature_scaled_0p5"] = (
            df["prob_temperature_scaled"] >= args.threshold).astype(int)

    metrics_rows = []
    bin_rows = []
    payload_metrics = {"val": {}, "test": {}}
    for split_name, df in (("val", val_df), ("test", test_df)):
        y = df["label"].to_numpy()
        for variant, column in (
            ("raw", "prob_raw"),
            ("temperature_scaled", "prob_temperature_scaled"),
        ):
            prob = df[column].to_numpy()
            metrics = metrics_for(y, prob, args.threshold, args.bins)
            payload_metrics[split_name][variant] = metrics
            row = {"split": split_name, "variant": variant, **metrics}
            metrics_rows.append(row)
            bin_rows.extend(build_bin_table(split_name, variant, y, prob, args.bins))

    metrics_df = pd.DataFrame(metrics_rows)
    bins_df = pd.DataFrame(bin_rows)
    test_out = test_df[[
        "ID", "SIDE", "label", "prob_raw", "prob_temperature_scaled",
        "pred_0p5", "pred_temperature_scaled_0p5",
    ]].copy()
    test_out.to_csv(calibrated_pred_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    bins_df.to_csv(bins_csv, index=False)

    payload = {
        "tag": args.tag,
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv),
        "temperature": temperature,
        "temperature_fit": {
            "split": "validation",
            "objective": "binary_nll",
            "note": "Fit to logits reconstructed from TTA-averaged probabilities.",
        },
        "threshold": args.threshold,
        "n_bins": args.bins,
        "metrics": payload_metrics,
        "artifacts": {
            "json": str(out_json),
            "metrics_csv": str(metrics_csv),
            "bins_csv": str(bins_csv),
            "calibrated_test_predictions": str(calibrated_pred_csv),
            "markdown": str(out_md),
            "figure": str(figure_path),
            "figure_pdf": str(figure_pdf_path),
            "caption": str(caption_path),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(out_md, payload)
    write_caption(caption_path, payload)
    plot_calibration_figure(
        figure_path, figure_pdf_path, bins_df, metrics_df, test_df, args.bins)

    test_raw = payload_metrics["test"]["raw"]
    test_temp = payload_metrics["test"]["temperature_scaled"]
    print("\nTEST CALIBRATION")
    print(f"Raw:  Brier {test_raw['brier']:.4f} | ECE {test_raw['ece']:.4f} | "
          f"NLL {test_raw['nll']:.4f}")
    print(f"Temp: Brier {test_temp['brier']:.4f} | ECE {test_temp['ece']:.4f} | "
          f"NLL {test_temp['nll']:.4f}")
    print(f"Saved JSON -> {out_json}")
    print(f"Saved metrics CSV -> {metrics_csv}")
    print(f"Saved bins CSV -> {bins_csv}")
    print(f"Saved figure -> {figure_path}")
    print(f"Saved figure PDF -> {figure_pdf_path}")
    print(f"Saved caption -> {caption_path}")


if __name__ == "__main__":
    main()
