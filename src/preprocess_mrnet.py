"""Preprocess extracted MRNet volumes into the 16-slice model layout."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

import config as cfg


PLANES = ("sagittal", "coronal", "axial")
PLANE_TO_OAI = {
    "sagittal": "dess",
    "coronal": "cor_mpr",
    "axial": "ax_mpr",
}
TASKS = ("abnormal", "acl", "meniscus")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess official MRNet valid split into OAI 16-slice layout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=cfg.MRNET_ROOT,
        help="Extracted MRNet-v1.0 dataset root containing train/valid folders.",
    )
    parser.add_argument("--split", choices=("valid", "train"), default="valid")
    parser.add_argument("--tag", default="mrnet_primary_16s_v1")
    parser.add_argument("--n_slices", type=int, default=16)
    parser.add_argument("--frac", type=float, default=0.80)
    parser.add_argument("--resize", type=int, default=cfg.RESIZE)
    parser.add_argument("--clip_low", type=float, default=cfg.CLIP_PERCENTILE[0])
    parser.add_argument("--clip_high", type=float, default=cfg.CLIP_PERCENTILE[1])
    parser.add_argument("--montage_cases", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_label_csv(root: Path, split: str, task: str) -> pd.Series:
    path = root / f"{split}-{task}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(
        path,
        header=None,
        names=["case_id", task],
        dtype={"case_id": str, task: int},
    )
    return df.set_index("case_id")[task]


def central_indices(n_slices: int, k: int, frac: float) -> list[int]:
    if n_slices <= 0:
        raise ValueError("Cannot sample from an empty volume")
    lo = int(round(n_slices * (1 - frac) / 2))
    hi = int(round(n_slices * (1 + frac) / 2)) - 1
    if hi <= lo:
        lo, hi = 0, max(n_slices - 1, 0)
    if k > 1:
        ids = [int(round(lo + (hi - lo) * i / (k - 1))) for i in range(k)]
    else:
        ids = [(lo + hi) // 2]
    return [min(max(idx, 0), n_slices - 1) for idx in ids]


def normalize_volume(vol: np.ndarray, clip_low: float, clip_high: float) -> tuple[np.ndarray, dict]:
    vol = vol.astype(np.float32, copy=False)
    flat = vol.reshape(-1)
    vmin = float(np.percentile(flat, clip_low))
    vmax = float(np.percentile(flat, clip_high))
    rng = (vmax - vmin) or 1.0
    norm = np.clip((vol - vmin) / rng, 0.0, 1.0)
    stats = {
        "clip_low": vmin,
        "clip_high": vmax,
        "raw_min": float(np.min(vol)),
        "raw_max": float(np.max(vol)),
        "raw_mean": float(np.mean(vol)),
        "raw_std": float(np.std(vol)),
    }
    return norm, stats


def save_png(slice_2d: np.ndarray, path: Path, resize: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((slice_2d * 255.0).round().astype(np.uint8))
    if img.size != (resize, resize):
        img = img.resize((resize, resize), Image.BILINEAR)
    img.save(path)


def safe_to_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_write_json(payload: dict, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_montages(index: pd.DataFrame, fig_dir: Path, n_cases: int,
                  n_slices: int, overwrite: bool) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    pos = index[index["label"] == 1].head(n_cases // 2)
    neg = index[index["label"] == 0].head(n_cases - len(pos))
    rows = pd.concat([pos, neg], axis=0).reset_index(drop=True)
    if rows.empty:
        return outputs

    for oai_plane in ("dess", "cor_mpr", "ax_mpr"):
        fig_rows = len(rows)
        fig, axes = plt.subplots(
            fig_rows,
            n_slices,
            figsize=(n_slices * 1.35, max(fig_rows, 1) * 1.45),
            dpi=150,
        )
        axes = np.asarray(axes).reshape(fig_rows, n_slices)
        for r_idx, (_, row) in enumerate(rows.iterrows()):
            for s_idx in range(n_slices):
                ax = axes[r_idx, s_idx]
                img = Image.open(row[f"{oai_plane}_{s_idx:02d}"]).convert("L")
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.axis("off")
                if r_idx == 0:
                    ax.set_title(str(s_idx), fontsize=7)
            axes[r_idx, 0].set_ylabel(
                f"{row['ID']} | y={int(row['label'])}",
                fontsize=7,
                rotation=0,
                labelpad=30,
                va="center",
            )
        fig.suptitle(f"MRNet external 9-slice preprocessing: {oai_plane}", fontsize=10)
        fig.tight_layout()
        out = fig_dir / f"{oai_plane}_selected_9s_montage.png"
        if out.exists() and not overwrite:
            raise SystemExit(f"Refusing to overwrite existing output: {out}")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(out))
    return outputs


def main():
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(args.root)

    out_dir = cfg.DATA / args.tag
    index_csv = cfg.DATA / f"{args.tag}_index.csv"
    manifest_csv = cfg.RESULTS / "logs" / f"{args.tag}_preprocess_manifest.csv"
    summary_json = cfg.RESULTS / "logs" / f"{args.tag}_preprocess_summary.json"
    protocol_md = cfg.RESULTS / "logs" / f"{args.tag}_preprocess_protocol.md"
    fig_dir = cfg.PROJECT / "figures" / f"{args.tag}_preprocess_qc"

    planned_outputs = [index_csv, manifest_csv, summary_json, protocol_md]
    for path in planned_outputs:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing output: {path}")
    if out_dir.exists() and any(out_dir.rglob("*.png")) and not args.overwrite:
        raise SystemExit(f"Refusing to write into non-empty image output: {out_dir}")

    labels = {task: read_label_csv(args.root, args.split, task) for task in TASKS}
    case_ids = sorted(labels["meniscus"].index.astype(str).tolist())

    index_rows = []
    manifest_rows = []
    for case_id in case_ids:
        row = {
            "ID": case_id,
            "SIDE": 0,
            "label": int(labels["meniscus"].loc[case_id]),
        }
        case_dir = out_dir / args.split / case_id
        for plane in PLANES:
            src_path = args.root / args.split / plane / f"{case_id}.npy"
            if not src_path.exists():
                raise FileNotFoundError(src_path)
            vol = np.load(src_path, mmap_mode=None, allow_pickle=False)
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D MRNet array, got {vol.shape}: {src_path}")
            norm, stats = normalize_volume(vol, args.clip_low, args.clip_high)
            selected = central_indices(norm.shape[0], args.n_slices, args.frac)
            oai_plane = PLANE_TO_OAI[plane]
            out_paths = []
            for out_idx, src_idx in enumerate(selected):
                out_path = case_dir / f"{oai_plane}_{out_idx:02d}.png"
                save_png(norm[src_idx], out_path, args.resize, args.overwrite)
                row[f"{oai_plane}_{out_idx:02d}"] = str(out_path)
                out_paths.append(str(out_path))
            manifest_rows.append({
                "split": args.split,
                "case_id": case_id,
                "source_plane": plane,
                "model_plane": oai_plane,
                "source_path": str(src_path),
                "n_source_slices": int(vol.shape[0]),
                "height": int(vol.shape[1]),
                "width": int(vol.shape[2]),
                "selected_indices": ";".join(str(idx) for idx in selected),
                "has_duplicate_indices": int(len(set(selected)) != len(selected)),
                "abnormal_label": int(labels["abnormal"].loc[case_id]),
                "acl_label": int(labels["acl"].loc[case_id]),
                "meniscus_label": int(labels["meniscus"].loc[case_id]),
                "output_paths": ";".join(out_paths),
                **stats,
            })
        index_rows.append(row)

    ordered_cols = (
        ["ID", "SIDE", "label"]
        + [f"{seq}_{idx:02d}" for seq in cfg.SEQUENCES for idx in range(args.n_slices)]
    )
    index = pd.DataFrame(index_rows).reindex(columns=ordered_cols)
    manifest = pd.DataFrame(manifest_rows)

    missing_paths = int(index[ordered_cols[3:]].isna().sum().sum())
    if missing_paths:
        raise RuntimeError(f"Internal error: {missing_paths} missing output paths")
    for col in ordered_cols[3:]:
        missing = [path for path in index[col].astype(str) if not Path(path).exists()]
        if missing:
            raise RuntimeError(f"Missing generated image for {col}: {missing[:3]}")

    safe_to_csv(index, index_csv, args.overwrite)
    safe_to_csv(manifest, manifest_csv, args.overwrite)
    montage_paths = save_montages(index, fig_dir, args.montage_cases,
                                  args.n_slices, args.overwrite)

    per_plane = manifest.groupby("model_plane")["n_source_slices"].agg(
        ["count", "min", "median", "max", "mean"]
    ).reset_index()
    summary = {
        "status": (
            "PREPROCESSING_ONLY. MRNet was converted into a model-facing "
            "9-slice PNG index. No checkpoint inference or external validation "
            "has been performed."
        ),
        "source_root": str(args.root),
        "split_processed": args.split,
        "external_validation_split": "valid",
        "tag": args.tag,
        "case_count": int(len(index)),
        "positive_cases": int(index["label"].sum()),
        "negative_cases": int((1 - index["label"]).sum()),
        "n_slices_per_plane": int(args.n_slices),
        "sampling_fraction": float(args.frac),
        "clip_percentiles": [float(args.clip_low), float(args.clip_high)],
        "resize": int(args.resize),
        "plane_mapping": PLANE_TO_OAI,
        "source_slice_summary_by_model_plane": per_plane.to_dict(orient="records"),
        "duplicate_selected_index_rows": int(manifest["has_duplicate_indices"].sum()),
        "outputs": {
            "index_csv": str(index_csv),
            "manifest_csv": str(manifest_csv),
            "summary_json": str(summary_json),
            "protocol_md": str(protocol_md),
            "figures_dir": str(fig_dir),
            "montages": montage_paths,
        },
        "clean_external_validation_notes": [
            "Only the official MRNet valid split is processed by default.",
            "Sampling and normalization are label-blind.",
            "MRNet train is not used for threshold fitting or calibration.",
            "The model-facing CSV mirrors data/processed_index_9s.csv columns.",
            "A separate external-validation script is still required; normal OAI loaders merge OAI MOAKS labels and splits.",
        ],
    }
    safe_write_json(summary, summary_json, args.overwrite)

    protocol = f"""# MRNet 16-Slice External Preprocessing Protocol

Status: preprocessing only. No external model evaluation was performed.

- Source: `{args.root}`
- Split processed: `{args.split}`
- Cases: {len(index)}
- Meniscus positives/negatives: {int(index["label"].sum())}/{int((1 - index["label"]).sum())}
- Plane mapping: sagittal -> dess, coronal -> cor_mpr, axial -> ax_mpr
- Slice sampling: {args.n_slices} ordered slices from the central {args.frac:.2f} of each volume
- Intensity normalization: per-volume percentile clipping at {args.clip_low:g}/{args.clip_high:g}, scaled to uint8 PNG
- Output image size: {args.resize}x{args.resize}
- Model-facing index: `{index_csv}`
- Sidecar manifest: `{manifest_csv}`

Clean external-validation constraint: do not use MRNet train for threshold
selection or calibration if reporting MRNet valid as a pure external validation
set.
"""
    if protocol_md.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {protocol_md}")
    protocol_md.write_text(protocol, encoding="utf-8")

    print("=== MRNet 9-slice external preprocessing ===")
    print(f"Source root : {args.root}")
    print(f"Split       : {args.split}")
    print(f"Cases       : {len(index)}")
    print(f"Pos/Neg     : {int(index['label'].sum())}/{int((1 - index['label']).sum())}")
    print(f"Index       : {index_csv}")
    print(f"Manifest    : {manifest_csv}")
    print(f"Summary     : {summary_json}")
    print(f"Protocol    : {protocol_md}")
    for path in montage_paths:
        print(f"Montage     : {path}")


if __name__ == "__main__":
    main()
