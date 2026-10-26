"""Derive binary tear and severity labels from MOAKS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

import config as cfg
from knee_utils import dedupe_knee_rows


def main() -> None:
    cfg.ensure_dirs()

    # OAI SAS-exported files contain stray high bytes (e.g. \xb4) in free-text
    # comment fields; read with latin1 which maps every byte 1:1 to Unicode.
    print("Reading MOAKS baseline:", cfg.MOAKS_TXT)
    df = pd.read_csv(cfg.MOAKS_TXT, sep="|", dtype=str, encoding="latin1")
    print("  raw shape:", df.shape)

    keep = ["ID", "SIDE"] + cfg.TEAR_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")

    sub = df[keep].copy()
    sub["ID"] = sub["ID"].str.strip()

    # Some OAI SAS-exported columns embed value labels, e.g. "1: Right" or
    # "3: Horizontal tear". Extract the leading integer code (NaN if none).
    def first_int(x):
        x = str(x).strip()
        return int(x.split(":")[0]) if x and x.split(":")[0].lstrip("-").isdigit() else np.nan

    sub["SIDE"] = sub["SIDE"].map(first_int).astype("Int64")

    # Coerce the six tear columns to numeric (e.g. "3: Horizontal tear" -> 3;
    # SAS '.' / blank -> NaN).
    for c in cfg.TEAR_COLS:
        sub[c] = sub[c].map(first_int)

    # Derive binary label: any region >= 2.
    tear_block = sub[cfg.TEAR_COLS]
    sub["label"] = (tear_block.fillna(0) >= 2).any(axis=1).astype(int)

    # Rename V00MMTMA -> MMTMA etc. for compactness.
    sub = sub.rename(columns=cfg.RENAME_COLS)
    sub = sub[["ID", "SIDE", "label"] + list(cfg.RENAME_COLS.values())]
    dup_groups = int(sub.groupby(["ID", "SIDE"]).size().gt(1).sum())
    sub = dedupe_knee_rows(sub)

    # --- report -----------------------------------------------------------
    n = len(sub)
    n_pos = int(sub["label"].sum())
    n_neg = n - n_pos
    print("\n=== Label balance (knee-level) ===")
    print(f"  total knees   : {n}")
    print(f"  tear (1)      : {n_pos}  ({n_pos/n:.1%})")
    print(f"  no-tear (0)   : {n_neg}  ({n_neg/n:.1%})")
    print(f"  unique IDs    : {sub['ID'].nunique()}")
    print(f"  side R/L      : {(sub['SIDE']==cfg.SIDE_RIGHT).sum()} / "
          f"{(sub['SIDE']==cfg.SIDE_LEFT).sum()}")
    print(f"  duplicate knees collapsed : {dup_groups}")

    # Per-region tear prevalence (score >= 2).
    print("\n=== Per-region tear prevalence (score >= 2) ===")
    region_cols = list(cfg.RENAME_COLS.values())
    region_names = {
        "MMTMA": "Medial anterior", "MMTMB": "Medial body", "MMTMP": "Medial posterior",
        "MMTLA": "Lateral anterior", "MMTLB": "Lateral body", "MMTLP": "Lateral posterior",
    }
    for c in region_cols:
        n_tear = int((sub[c] >= 2).sum())
        print(f"  {c} ({region_names[c]:18s}): {n_tear:4d} ({n_tear/n:.2%})")

    sub.to_csv(cfg.LABELS_CSV, index=False)
    print(f"\nSaved -> {cfg.LABELS_CSV}")


if __name__ == "__main__":
    main()
