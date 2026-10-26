"""Extended bootstrap summaries for the primary model."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as cfg
from bootstrap_performance_ci import align_pair, guard_outputs, load_predictions

RESULTS = cfg.RESULTS

DEFAULT_CONVNEXT = RESULTS / "convnext_safe_strong_16s_tta_20260629_coverage_test_predictions.csv"
DEFAULT_RESNET = RESULTS / "primary_model_safe_strong_16s_tta_coverage_20260628_test_predictions.csv"
DEFAULT_CALIB = RESULTS / "convnext_safe_strong_16s_tta_20260629_calibration_test_predictions.csv"
DEFAULT_CONFORMAL = RESULTS / "convnext_safe_strong_16s_tta_20260629_conformal_test_predictions.csv"
DEFAULT_COVERAGE_METRICS = RESULTS / "metrics" / "convnext_safe_strong_16s_tta_20260629_coverage.csv"
DEFAULT_MRNET = RESULTS / "metrics" / "mrnet_external_v2_predictions.csv"
DEFAULT_LOCATION = RESULTS / "secondary_location_9s_convnext_audit_20260629_v1_test_location_predictions.csv"

DEFAULT_TAG = "convnext_safe_strong_16s_tta_20260629_extended_ci"
COVERAGE_TARGETS = (0.9, 0.8, 0.7)
CONF_ALPHAS = (0.05, 0.1, 0.2)
N_ECE_BINS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended bootstrap 95% CIs for calibration, selective, "
                    "conformal, MRNet, and severity metrics")
    parser.add_argument("--convnext_csv", type=Path, default=DEFAULT_CONVNEXT)
    parser.add_argument("--resnet_csv", type=Path, default=DEFAULT_RESNET)
    parser.add_argument("--calib_csv", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--conformal_csv", type=Path, default=DEFAULT_CONFORMAL)
    parser.add_argument("--coverage_metrics_csv", type=Path,
                        default=DEFAULT_COVERAGE_METRICS)
    parser.add_argument("--mrnet_csv", type=Path, default=DEFAULT_MRNET)
    parser.add_argument("--location_csv", type=Path, default=DEFAULT_LOCATION)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_calibrated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "SIDE", "label", "prob_raw", "prob_temperature_scaled"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["ID"] = df["ID"].astype(str)
    df["SIDE"] = df["SIDE"].astype(int)
    df["label"] = df["label"].astype(int)
    df["prob_temperature_scaled"] = (
        df["prob_temperature_scaled"].astype(float).clip(1e-6, 1 - 1e-6))
    return df[["ID", "SIDE", "label", "prob_temperature_scaled"]] \
        .reset_index(drop=True)


def load_conformal_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str})
    df = df[df["split"] == "test"].copy()
    df["ID"] = df["ID"].astype(str)
    df["SIDE"] = df["SIDE"].astype(int)
    df["alpha"] = df["alpha"].astype(float).round(2)
    pieces = []
    for alpha in CONF_ALPHAS:
        sub = df[df["alpha"] == alpha]
        if sub.empty:
            raise ValueError(f"conformal csv missing alpha={alpha}")
        piece = sub[["ID", "SIDE", "true_label_covered", "singleton_pred",
                     "correct_singleton"]].copy()
        piece = piece.rename(columns={
            "true_label_covered": f"covered_{alpha}",
            "singleton_pred": f"single_{alpha}",
            "correct_singleton": f"corr_single_{alpha}",
        })
        dup = piece.duplicated(subset=["ID", "SIDE"])
        if dup.any():
            raise ValueError(f"duplicate knees for alpha={alpha}")
        pieces.append(piece)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=["ID", "SIDE"], how="inner",
                        validate="one_to_one")
    return out.reset_index(drop=True)


def load_location_severity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "SIDE", "severity_true", "severity_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["ID"] = df["ID"].astype(str)
    df["SIDE"] = df["SIDE"].astype(int)
    df["severity_true"] = df["severity_true"].astype(int)
    df["severity_pred"] = df["severity_pred"].astype(int)
    return df[["ID", "SIDE", "severity_true", "severity_pred"]] \
        .reset_index(drop=True)


def load_val_cutoffs(path: Path) -> dict:
    df = pd.read_csv(path)
    df = df[(df["split"] == "val") & (df["target_coverage"].isin(COVERAGE_TARGETS))]
    cutoffs = {}
    for _, row in df.iterrows():
        cutoffs[float(row["target_coverage"])] = float(row["confidence_cutoff"])
    missing = set(COVERAGE_TARGETS).difference(cutoffs)
    if missing:
        raise ValueError(f"validation cutoffs missing for targets: {missing}")
    return cutoffs


def ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(y)
    value = 0.0
    for b in range(n_bins):
        mask = indices == b
        if not mask.any():
            continue
        value += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def calibration_metrics(y: np.ndarray, p_t: np.ndarray) -> dict:
    eps = 1e-6
    pc = np.clip(p_t, eps, 1 - eps)
    nll = float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
    brier = float(np.mean((p_t - y) ** 2))
    return {
        "brier_T": brier,
        "nll_T": nll,
        "ece10_T": ece_binary(y, p_t),
    }


def selective_metrics(y: np.ndarray, p: np.ndarray, cutoff: float,
                      threshold: float) -> dict:
    confidence = np.maximum(p, 1.0 - p)
    selected = confidence >= cutoff
    n_sel = int(selected.sum())
    out = {
        "coverage": float(n_sel / len(y)) if len(y) else np.nan,
        "auc_selected": np.nan,
        "error_rate_selected": np.nan,
    }
    if n_sel > 0:
        ys = y[selected]
        ps = p[selected]
        out["auc_selected"] = safe_auc(ys, ps)
        errors = (ps >= threshold).astype(int) != ys
        out["error_rate_selected"] = float(errors.mean())
    return out


def conformal_metrics(block: pd.DataFrame, alpha: float) -> dict:
    covered = pd.to_numeric(block[f"covered_{alpha}"], errors="coerce") == 1
    single = pd.to_numeric(block[f"single_{alpha}"], errors="coerce") == 1
    correct = pd.to_numeric(block[f"corr_single_{alpha}"], errors="coerce")
    prefix = f"conf{int(round(alpha * 100)):02d}"
    single_acc = correct[single].dropna()
    return {
        f"{prefix}_coverage": float(covered.mean()) if len(covered) else np.nan,
        f"{prefix}_singleton_rate": float(single.mean()) if len(single) else np.nan,
        f"{prefix}_singleton_accuracy": float(single_acc.mean())
        if len(single_acc) else np.nan,
        f"{prefix}_defer_rate": float(1.0 - single.mean())
        if len(single) else np.nan,
    }


def severity_metrics(block: pd.DataFrame) -> dict:
    y = block["severity_true"].to_numpy()
    pred = block["severity_pred"].to_numpy()
    acc = float((y == pred).mean()) if len(y) else np.nan
    recalls = []
    for cls in np.unique(y):
        mask = y == cls
        recalls.append(float((pred[mask] == cls).mean()))
    return {
        "severity_accuracy": acc,
        "severity_balanced_accuracy": float(np.mean(recalls))
        if recalls else np.nan,
    }


def build_master(convnext: pd.DataFrame, resnet: pd.DataFrame,
                 calib: pd.DataFrame, conformal: pd.DataFrame,
                 severity: pd.DataFrame) -> pd.DataFrame:
    pair = align_pair(convnext, resnet)
    master = pair.merge(calib.drop(columns=["label"]), on=["ID", "SIDE"],
                        how="inner", validate="one_to_one")
    master = master.merge(conformal, on=["ID", "SIDE"], how="inner",
                          validate="one_to_one")
    master = master.merge(severity, on=["ID", "SIDE"], how="inner",
                          validate="one_to_one")
    if len(master) != len(pair):
        raise ValueError(
            f"master alignment lost: pair={len(pair)} master={len(master)}")
    return master.reset_index(drop=True)


def point_estimates(master: pd.DataFrame, cutoffs: dict,
                    threshold: float) -> dict:
    y = master["label"].to_numpy()
    p_raw = master["prob_convnext"].to_numpy()
    p_res = master["prob_resnet"].to_numpy()
    p_t = master["prob_temperature_scaled"].to_numpy()

    values = {
        "auc": safe_auc(y, p_raw),
        "pr_auc": float(average_precision_score(y, p_raw)),
    }
    values.update(calibration_metrics(y, p_t))
    for target, cutoff in cutoffs.items():
        key = f"cov{int(round(target * 100)):02d}"
        sel = selective_metrics(y, p_raw, cutoff, threshold)
        values[f"{key}_achieved_coverage"] = sel["coverage"]
        values[f"{key}_auc_selected"] = sel["auc_selected"]
        values[f"{key}_error_rate_selected"] = sel["error_rate_selected"]
    for alpha in CONF_ALPHAS:
        values.update(conformal_metrics(master, alpha))
    values.update(severity_metrics(master))
    return values


def bootstrap_master(master: pd.DataFrame, cutoffs: dict, n_boot: int,
                     rng: np.random.Generator,
                     threshold: float) -> dict[str, np.ndarray]:
    point = point_estimates(master, cutoffs, threshold)
    arrays = {key: [] for key in point}
    ids = master["ID"].drop_duplicates().to_numpy()
    index_by_id = {pid: np.where(master["ID"].to_numpy() == pid)[0]
                   for pid in ids}
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([index_by_id[pid] for pid in sampled])
        block = master.iloc[idx].reset_index(drop=True)
        rep = point_estimates(block, cutoffs, threshold)
        for key in arrays:
            arrays[key].append(rep[key])
    return {key: np.asarray(val, dtype=float) for key, val in arrays.items()}, point


def bootstrap_mrnet(df: pd.DataFrame, prob_col: str, n_boot: int,
                    rng: np.random.Generator) -> tuple[dict, dict[str, np.ndarray]]:
    y_all = df["label"].astype(int).to_numpy()
    p_all = df[prob_col].astype(float).to_numpy()
    point = {"mrnet_auc": safe_auc(y_all, p_all)}
    ids = df["ID"].drop_duplicates().to_numpy()
    index_by_id = {pid: np.where(df["ID"].to_numpy() == pid)[0] for pid in ids}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([index_by_id[pid] for pid in sampled])
        values.append(safe_auc(y_all[idx], p_all[idx]))
    return point, {"mrnet_auc": np.asarray(values, dtype=float)}


def summarize_rows(family: str, point: dict,
                   arrays: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for metric, samples in arrays.items():
        finite = samples[np.isfinite(samples)]
        estimate = point.get(metric, np.nan)
        rows.append({
            "family": family,
            "metric": metric,
            "estimate": float(estimate) if estimate is not None else np.nan,
            "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
            "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
            "bootstrap_se": float(np.std(finite, ddof=1)) if len(finite) else np.nan,
            "n_boot_valid": int(len(finite)),
        })
    return rows


METRIC_LABELS = {
    "auc": "Test AUC (7x TTA)",
    "pr_auc": "Test PR-AUC (7x TTA)",
    "brier_T": "Brier after temperature scaling",
    "nll_T": "NLL after temperature scaling",
    "ece10_T": "ECE-10 after temperature scaling",
    "cov90_achieved_coverage": "Achieved coverage @90% target",
    "cov80_achieved_coverage": "Achieved coverage @80% target",
    "cov70_achieved_coverage": "Achieved coverage @70% target",
    "cov90_auc_selected": "AUC among auto-read knees @90% target",
    "cov80_auc_selected": "AUC among auto-read knees @80% target",
    "cov70_auc_selected": "AUC among auto-read knees @70% target",
    "cov90_error_rate_selected": "Error rate among auto-read knees @90% target",
    "cov80_error_rate_selected": "Error rate among auto-read knees @80% target",
    "cov70_error_rate_selected": "Error rate among auto-read knees @70% target",
    "conf05_coverage": "Conformal empirical coverage @95% target",
    "conf10_coverage": "Conformal empirical coverage @90% target",
    "conf20_coverage": "Conformal empirical coverage @80% target",
    "conf05_singleton_rate": "Singleton (auto-read) rate @95% target",
    "conf10_singleton_rate": "Singleton (auto-read) rate @90% target",
    "conf20_singleton_rate": "Singleton (auto-read) rate @80% target",
    "conf05_singleton_accuracy": "Accuracy among singleton sets @95% target",
    "conf10_singleton_accuracy": "Accuracy among singleton sets @90% target",
    "conf20_singleton_accuracy": "Accuracy among singleton sets @80% target",
    "conf05_defer_rate": "Defer rate @95% target",
    "conf10_defer_rate": "Defer rate @90% target",
    "conf20_defer_rate": "Defer rate @80% target",
    "severity_accuracy": "Secondary-model severity exact-match accuracy",
    "severity_balanced_accuracy": "Secondary-model severity balanced accuracy",
    "mrnet_auc_locked_valid": "MRNet locked-valid AUC (focused-pair rank avg)",
    "mrnet_auc_all_public": (
        "MRNet all-public zero-shot transfer AUC (focused-pair rank avg)"),
}


def write_markdown(path: Path, rows: list[dict], payload: dict):
    lines = [
        "# Extended Bootstrap Confidence Intervals",
        "",
        "- Bootstrap unit: participant ID; all knees of a sampled participant "
        "are kept together.",
        f"- Replicates: `{payload['n_boot']}`",
        "- All ConvNeXt metric families share the same participant resample "
        "within each replicate.",
        f"- Test knees: `{payload['n_test_knees']}` | test participants: "
        f"`{payload['n_test_participants']}`",
        "",
        "| Family | Metric | Estimate | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        label = METRIC_LABELS.get(row["metric"], row["metric"])
        lines.append(
            f"| {row['family']} | {label} | {row['estimate']:.4f} | "
            f"{row['ci_low']:.4f}-{row['ci_high']:.4f} |")
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    metrics_dir = RESULTS / "metrics"
    logs_dir = RESULTS / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / f"{args.tag}.json"
    out_csv = metrics_dir / f"{args.tag}_metrics.csv"
    out_md = logs_dir / f"{args.tag}.md"
    guard_outputs([out_json, out_csv, out_md], args.overwrite)

    cutoffs = load_val_cutoffs(args.coverage_metrics_csv)
    convnext = load_predictions(args.convnext_csv)
    resnet = load_predictions(args.resnet_csv)
    calib = load_calibrated(args.calib_csv)
    conformal = load_conformal_wide(args.conformal_csv)
    severity = load_location_severity(args.location_csv)
    master = build_master(convnext, resnet, calib, conformal, severity)

    rng = np.random.default_rng(args.seed)
    arrays, point = bootstrap_master(master, cutoffs, args.n_boot, rng,
                                     args.threshold)
    rows = summarize_rows("ConvNeXt internal test", point, arrays)

    mrnet_df = pd.read_csv(args.mrnet_csv, dtype={"ID": str})
    prob_col = "focused_pair_rank_average"
    for subset, family in (("valid", "MRNet locked-valid"),
                           (None, "MRNet all-public zero-shot transfer")):
        block = mrnet_df if subset is None else \
            mrnet_df[mrnet_df["split"] == subset]
        block = block.reset_index(drop=True)
        m_point, m_arrays = bootstrap_mrnet(block, prob_col, args.n_boot, rng)
        key = ("mrnet_auc_locked_valid" if subset == "valid"
               else "mrnet_auc_all_public")
        rows.extend(summarize_rows(
            family, {key: m_point["mrnet_auc"]},
            {key: m_arrays["mrnet_auc"]}))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    payload = {
        "tag": args.tag,
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "threshold": float(args.threshold),
        "bootstrap_unit": "participant ID",
        "n_test_knees": int(len(master)),
        "n_test_participants": int(master["ID"].nunique()),
        "inputs": {
            "convnext_csv": str(args.convnext_csv),
            "resnet_csv": str(args.resnet_csv),
            "calib_csv": str(args.calib_csv),
            "conformal_csv": str(args.conformal_csv),
            "coverage_metrics_csv": str(args.coverage_metrics_csv),
            "mrnet_csv": str(args.mrnet_csv),
            "location_csv": str(args.location_csv),
        },
        "artifacts": {
            "json": str(out_json),
            "metrics_csv": str(out_csv),
            "markdown": str(out_md),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(out_md, rows, payload)

    print(f"Saved extended metrics -> {out_csv}")
    print(f"Saved report -> {out_md}")


if __name__ == "__main__":
    main()
