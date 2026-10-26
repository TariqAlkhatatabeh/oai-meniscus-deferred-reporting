"""Conformal-style prediction sets for the primary model."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config as cfg


DEFAULT_TAG = "convnext_safe_strong_16s_tta_20260629_conformal"
DEFAULT_VAL = cfg.RESULTS / "convnext_safe_strong_16s_tta_20260629_coverage_val_predictions.csv"
DEFAULT_TEST = cfg.RESULTS / "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"
DEFAULT_CALIBRATION_JSON = (
    cfg.RESULTS / "metrics" /
    "convnext_safe_strong_16s_tta_20260629_calibration.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-locked conformal-style prediction sets")
    parser.add_argument("--val_csv", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test_csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--calibration_json", type=Path,
                        default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20],
                        help="Target error rates; coverage is 1-alpha.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def guard_outputs(paths, overwrite: bool):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing conformal outputs. Use a new "
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
    df["prob_raw"] = df["prob"].astype(float).clip(1e-6, 1 - 1e-6)
    return df


def load_temperature(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    temperature = float(payload["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid temperature in {path}: {temperature}")
    return temperature


def logit_np(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(prob / (1.0 - prob))


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def apply_temperature(prob, temperature: float):
    return sigmoid_np(logit_np(prob) / temperature)


def add_calibrated_probabilities(df: pd.DataFrame, temperature: float) -> pd.DataFrame:
    df = df.copy()
    df["prob_temperature_scaled"] = apply_temperature(
        df["prob_raw"].to_numpy(), temperature)
    return df


def true_label_scores(df: pd.DataFrame) -> np.ndarray:
    prob_tear = df["prob_temperature_scaled"].to_numpy()
    labels = df["label"].to_numpy()
    prob_true = np.where(labels == 1, prob_tear, 1.0 - prob_tear)
    return 1.0 - prob_true


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(scores[rank - 1])


def label_set_name(include_no_tear: bool, include_tear: bool) -> str:
    if include_no_tear and include_tear:
        return "ambiguous"
    if include_no_tear:
        return "no_tear"
    if include_tear:
        return "tear"
    return "empty"


def build_prediction_rows(df: pd.DataFrame, alpha: float, qhat: float) -> pd.DataFrame:
    out = df[["ID", "SIDE", "label", "prob_raw",
              "prob_temperature_scaled"]].copy()
    p = out["prob_temperature_scaled"].to_numpy()
    include_no_tear = p <= qhat
    include_tear = (1.0 - p) <= qhat

    out["alpha"] = alpha
    out["target_coverage"] = 1.0 - alpha
    out["qhat"] = qhat
    out["include_no_tear"] = include_no_tear.astype(int)
    out["include_tear"] = include_tear.astype(int)
    out["set_size"] = out["include_no_tear"] + out["include_tear"]
    out["label_set"] = [
        label_set_name(bool(n), bool(t))
        for n, t in zip(include_no_tear, include_tear)
    ]
    out["clinical_action"] = out["label_set"].replace({
        "no_tear": "predict_no_tear",
        "tear": "predict_tear",
        "ambiguous": "defer",
        "empty": "defer",
    })
    out["singleton_pred"] = np.nan
    out.loc[out["label_set"] == "no_tear", "singleton_pred"] = 0
    out.loc[out["label_set"] == "tear", "singleton_pred"] = 1
    out["true_label_covered"] = (
        ((out["label"] == 0) & (out["include_no_tear"] == 1)) |
        ((out["label"] == 1) & (out["include_tear"] == 1))
    ).astype(int)
    out["correct_singleton"] = np.where(
        out["set_size"] == 1,
        (out["singleton_pred"].astype(float) == out["label"].astype(float)).astype(int),
        np.nan,
    )
    return out


def binary_metrics_for_singletons(sub: pd.DataFrame):
    singleton = sub[sub["set_size"] == 1].copy()
    if singleton.empty:
        return {
            "singleton_accuracy": np.nan,
            "singleton_f1": np.nan,
            "singleton_sensitivity": np.nan,
            "singleton_specificity": np.nan,
            "singleton_error_rate": np.nan,
        }

    y_true = singleton["label"].astype(int).to_numpy()
    y_pred = singleton["singleton_pred"].astype(int).to_numpy()
    accuracy = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return {
        "singleton_accuracy": accuracy,
        "singleton_f1": f1,
        "singleton_sensitivity": sensitivity,
        "singleton_specificity": specificity,
        "singleton_error_rate": float(1.0 - accuracy),
    }


def summarize_split(pred_rows: pd.DataFrame, split: str) -> dict:
    n = int(len(pred_rows))
    singleton = pred_rows[pred_rows["set_size"] == 1]
    ambiguous = pred_rows[pred_rows["label_set"] == "ambiguous"]
    empty = pred_rows[pred_rows["label_set"] == "empty"]
    no_tear = pred_rows[pred_rows["label_set"] == "no_tear"]
    tear = pred_rows[pred_rows["label_set"] == "tear"]

    label0 = pred_rows[pred_rows["label"] == 0]
    label1 = pred_rows[pred_rows["label"] == 1]
    metrics = {
        "split": split,
        "alpha": float(pred_rows["alpha"].iloc[0]),
        "target_coverage": float(pred_rows["target_coverage"].iloc[0]),
        "qhat": float(pred_rows["qhat"].iloc[0]),
        "n": n,
        "empirical_coverage": float(pred_rows["true_label_covered"].mean()),
        "class0_coverage": (
            float(label0["true_label_covered"].mean()) if len(label0) else np.nan
        ),
        "class1_coverage": (
            float(label1["true_label_covered"].mean()) if len(label1) else np.nan
        ),
        "singleton_n": int(len(singleton)),
        "singleton_rate": float(len(singleton) / n),
        "defer_n": int(len(ambiguous) + len(empty)),
        "defer_rate": float((len(ambiguous) + len(empty)) / n),
        "ambiguous_n": int(len(ambiguous)),
        "ambiguous_rate": float(len(ambiguous) / n),
        "empty_n": int(len(empty)),
        "empty_rate": float(len(empty) / n),
        "singleton_no_tear_n": int(len(no_tear)),
        "singleton_tear_n": int(len(tear)),
    }
    metrics.update(binary_metrics_for_singletons(pred_rows))
    return metrics


def format_percent(value):
    if pd.isna(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def write_markdown(path: Path, payload: dict, metrics_df: pd.DataFrame):
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values("target_coverage")
    lines = [
        "# Primary Model Conformal-Style Prediction Sets",
        "",
        f"- Tag: `{payload['tag']}`",
        f"- Temperature: `{payload['temperature']:.4f}`",
        "- Method: binary split-conformal style label sets using validation "
        "nonconformity scores `1 - P_temperature_scaled(true label)`.",
        "- Reporting note: the validation set was also used for checkpoint "
        "selection and temperature scaling, so this is best described as a "
        "validation-locked conformal-style/defer analysis unless a dedicated "
        "calibration split is created later.",
        "- Ambiguous `{no tear, tear}` and empty `{}` sets are both counted as "
        "`defer` for clinical reporting.",
        "",
        "## Test Results",
        "",
        "| Target coverage | qhat | Empirical coverage | Singleton rate | "
        "Defer rate | Singleton accuracy | Singleton F1 | Sens | Spec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_df.to_dict("records"):
        lines.append(
            f"| {format_percent(row['target_coverage'])} | "
            f"{row['qhat']:.4f} | "
            f"{format_percent(row['empirical_coverage'])} | "
            f"{format_percent(row['singleton_rate'])} | "
            f"{format_percent(row['defer_rate'])} | "
            f"{format_percent(row['singleton_accuracy'])} | "
            f"{format_percent(row['singleton_f1'])} | "
            f"{format_percent(row['singleton_sensitivity'])} | "
            f"{format_percent(row['singleton_specificity'])} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def write_caption(path: Path, payload: dict, metrics_df: pd.DataFrame):
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values(
        "target_coverage")
    parts = []
    for row in test_df.to_dict("records"):
        parts.append(
            f"{format_percent(row['target_coverage'])} target: "
            f"{format_percent(row['empirical_coverage'])} empirical coverage, "
            f"{format_percent(row['singleton_rate'])} singleton outputs"
        )
    text = (
        "Figure. Validation-locked conformal-style uncertainty analysis for the "
        "locked 16-slice safe-strong model with 7x test-time augmentation and "
        "temperature-scaled probabilities. (A) Empirical test coverage compared "
        "with the nominal target coverage. (B) Test prediction-set composition, "
        "including singleton no-tear, singleton tear, ambiguous, and empty/defer "
        "outputs. (C) Performance among singleton predictions only. (D) "
        "Temperature-scaled probability distribution with the 90% target "
        "conformal decision regions. " + " ".join(parts) + "."
    )
    path.write_text(text + "\n")


def plot_conformal_figure(path: Path, pdf_path: Path, metrics_df: pd.DataFrame,
                          long_test: pd.DataFrame):
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

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values("target_coverage")
    x_labels = [f"{int(round(v * 100))}%" for v in test_metrics["target_coverage"]]
    x = np.arange(len(test_metrics))

    fig = plt.figure(figsize=(8.2, 6.2), dpi=300)
    grid = GridSpec(2, 2, figure=fig, wspace=0.52, hspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    ax_a.plot(x, test_metrics["target_coverage"], marker="o", color="0.35",
              linestyle="--", linewidth=1.3, label="Target")
    ax_a.plot(x, test_metrics["empirical_coverage"], marker="o",
              color="#2f6fbb", linewidth=1.7, label="Observed")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x_labels)
    ax_a.set_ylim(0.72, 1.02)
    ax_a.set_xlabel("Nominal coverage")
    ax_a.set_ylabel("True-label coverage")
    ax_a.set_title("Coverage")
    ax_a.legend(frameon=False, loc="lower right")
    ax_a.grid(True, linestyle=":", alpha=0.45)

    composition = [
        ("singleton_no_tear_n", "No-tear singleton", "#4c78a8"),
        ("singleton_tear_n", "Tear singleton", "#f58518"),
        ("ambiguous_n", "Ambiguous", "#72b7b2"),
        ("empty_n", "Empty/defer", "#b279a2"),
    ]
    bottom = np.zeros(len(test_metrics), dtype=float)
    for column, label, color in composition:
        values = test_metrics[column].to_numpy(dtype=float) / test_metrics["n"].to_numpy()
        ax_b.bar(x, values, bottom=bottom, width=0.62, label=label,
                 color=color, alpha=0.9)
        bottom += values
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(x_labels)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("Nominal coverage")
    ax_b.set_ylabel("Fraction of test set")
    ax_b.set_title("Prediction-set composition")
    ax_b.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.45)

    perf_cols = [
        ("singleton_accuracy", "Accuracy", "#2f6fbb"),
        ("singleton_f1", "F1", "#d95f02"),
        ("singleton_sensitivity", "Sensitivity", "#1b9e77"),
        ("singleton_specificity", "Specificity", "#7570b3"),
    ]
    for col, label, color in perf_cols:
        ax_c.plot(x, test_metrics[col], marker="o", linewidth=1.5,
                  label=label, color=color)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(x_labels)
    ax_c.set_ylim(0.55, 1.02)
    ax_c.set_xlabel("Nominal coverage")
    ax_c.set_ylabel("Metric among singletons")
    ax_c.set_title("Singleton performance")
    ax_c.legend(frameon=False, loc="lower right")
    ax_c.grid(True, linestyle=":", alpha=0.45)

    alpha_90 = sorted(long_test["alpha"].unique(),
                      key=lambda a: abs(float(a) - 0.10))[0]
    sub90 = long_test[long_test["alpha"] == alpha_90].copy()
    qhat = float(sub90["qhat"].iloc[0])
    p_left = min(qhat, 1.0 - qhat)
    p_right = max(qhat, 1.0 - qhat)
    bins = np.linspace(0, 1, 21)
    neg = sub90.loc[sub90["label"] == 0, "prob_temperature_scaled"]
    pos = sub90.loc[sub90["label"] == 1, "prob_temperature_scaled"]
    ax_d.hist(neg, bins=bins, alpha=0.72, color="#4c78a8",
              label="No tear")
    ax_d.hist(pos, bins=bins, alpha=0.72, color="#f58518",
              label="Tear")
    ax_d.axvline(p_left, color="0.2", linestyle="--", linewidth=1)
    ax_d.axvline(p_right, color="0.2", linestyle="--", linewidth=1)
    if qhat >= 0.5:
        ax_d.axvspan(p_left, p_right, color="#72b7b2", alpha=0.14,
                     label="Ambiguous region")
    else:
        ax_d.axvspan(p_left, p_right, color="#b279a2", alpha=0.14,
                     label="Empty/defer region")
    ax_d.set_xlabel("Temperature-scaled tear probability")
    ax_d.set_ylabel("Number of knees")
    ax_d.set_title("90% target regions")
    ax_d.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0)
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
    alphas = sorted(set(float(a) for a in args.alphas))
    metrics_dir = cfg.RESULTS / "metrics"
    logs_dir = cfg.RESULTS / "logs"
    figure_dir = cfg.PROJECT / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / f"{args.tag}.json"
    metrics_csv = metrics_dir / f"{args.tag}_metrics.csv"
    test_predictions_csv = cfg.RESULTS / f"{args.tag}_test_predictions.csv"
    val_scores_csv = metrics_dir / f"{args.tag}_validation_scores.csv"
    out_md = logs_dir / f"{args.tag}.md"
    caption_path = logs_dir / f"{args.tag}_figure_caption.txt"
    figure_path = figure_dir / f"{args.tag}.png"
    figure_pdf_path = figure_dir / f"{args.tag}.pdf"
    guard_outputs(
        [out_json, metrics_csv, test_predictions_csv, val_scores_csv,
         out_md, caption_path, figure_path, figure_pdf_path],
        args.overwrite,
    )

    temperature = load_temperature(args.calibration_json)
    val_df = add_calibrated_probabilities(load_predictions(args.val_csv), temperature)
    test_df = add_calibrated_probabilities(load_predictions(args.test_csv), temperature)
    val_scores = true_label_scores(val_df)

    val_score_df = val_df[["ID", "SIDE", "label", "prob_raw",
                           "prob_temperature_scaled"]].copy()
    val_score_df["nonconformity_score"] = val_scores
    val_score_df.to_csv(val_scores_csv, index=False)

    long_rows = []
    metric_rows = []
    qhats = {}
    for alpha in alphas:
        qhat = conformal_quantile(val_scores, alpha)
        qhats[str(alpha)] = qhat
        for split_name, df in (("val", val_df), ("test", test_df)):
            rows = build_prediction_rows(df, alpha, qhat)
            rows.insert(0, "split", split_name)
            metric_rows.append(summarize_split(rows, split_name))
            if split_name == "test":
                long_rows.append(rows)

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["split", "target_coverage"])
    long_test = pd.concat(long_rows, ignore_index=True)

    metrics_df.to_csv(metrics_csv, index=False)
    long_test.to_csv(test_predictions_csv, index=False)

    payload = {
        "tag": args.tag,
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv),
        "calibration_json": str(args.calibration_json),
        "temperature": temperature,
        "alphas": alphas,
        "target_coverages": [1.0 - a for a in alphas],
        "quantiles": qhats,
        "method": {
            "nonconformity_score": "1 - P_temperature_scaled(true_label)",
            "prediction_rule": (
                "include candidate label y if 1 - P_temperature_scaled(y) <= qhat"
            ),
            "defer_definition": "ambiguous and empty prediction sets",
            "caveat": (
                "Validation split was also used for checkpoint selection and "
                "temperature scaling; report as validation-locked "
                "conformal-style analysis unless a dedicated calibration split "
                "is created."
            ),
        },
        "artifacts": {
            "json": str(out_json),
            "metrics_csv": str(metrics_csv),
            "test_predictions_csv": str(test_predictions_csv),
            "validation_scores_csv": str(val_scores_csv),
            "markdown": str(out_md),
            "caption": str(caption_path),
            "figure": str(figure_path),
            "figure_pdf": str(figure_pdf_path),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(out_md, payload, metrics_df)
    write_caption(caption_path, payload, metrics_df)
    plot_conformal_figure(figure_path, figure_pdf_path, metrics_df, long_test)

    print("\nTEST CONFORMAL-STYLE RESULTS")
    test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values(
        "target_coverage")
    for row in test_metrics.to_dict("records"):
        print(
            f"Target {100 * row['target_coverage']:.0f}% | "
            f"q={row['qhat']:.4f} | "
            f"emp_cov={100 * row['empirical_coverage']:.1f}% | "
            f"singleton={100 * row['singleton_rate']:.1f}% | "
            f"defer={100 * row['defer_rate']:.1f}% | "
            f"singleton_acc={100 * row['singleton_accuracy']:.1f}%"
        )
    print(f"Saved JSON -> {out_json}")
    print(f"Saved metrics CSV -> {metrics_csv}")
    print(f"Saved test predictions -> {test_predictions_csv}")
    print(f"Saved figure -> {figure_path}")
    print(f"Saved figure PDF -> {figure_pdf_path}")
    print(f"Saved caption -> {caption_path}")


if __name__ == "__main__":
    main()
