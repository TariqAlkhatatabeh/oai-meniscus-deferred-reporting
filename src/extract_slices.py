"""Extract and normalize the 16-slice OAI representation."""
import argparse
import os
import re
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import SimpleITK as sitk

import config as cfg
from knee_utils import dedupe_knee_rows

# SeriesDescription matchers + side tokens, module-level for pickle to workers.
MATCHERS = {k: re.compile(v, re.IGNORECASE)
            for k, v in cfg.SEQ_MATCHERS.items()}
SIDE_TOKEN = {cfg.SIDE_RIGHT: "RIGHT", cfg.SIDE_LEFT: "LEFT"}

OUT_COLS = (["ID", "SIDE", "label"]
            + [f"{s}_{i:02d}" for s in cfg.SEQUENCES
               for i in range(cfg.N_SLICES)])
EXPECTED_COLUMNS = OUT_COLS


# --------------------------------------------------------------------------- #
# Per-participant worker
# --------------------------------------------------------------------------- #
def _first_dicom_header(series_dir: Path):
    """Read SeriesDescription from the first DICOM in series_dir."""
    import pydicom
    for fp in sorted(series_dir.iterdir()):
        if not fp.is_file() or fp.name == ".complete":
            continue
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            sd = getattr(ds, "SeriesDescription", "")
            if sd:
                return str(sd)
        except Exception:
            continue
    return ""


def _find_targets(pid_dir: Path, side: int) -> dict:
    """Map each target sequence -> series dir under pid_dir for the side."""
    token = SIDE_TOKEN.get(side)
    found = {k: None for k in cfg.SEQUENCES}
    if token is None:
        return found
    for series_dir in sorted(d for d in pid_dir.iterdir() if d.is_dir()):
        sd = _first_dicom_header(series_dir)
        if not sd:
            continue
        up = sd.upper()
        # Side filter: the target sequences encode side as _LEFT/_RIGHT.
        if SIDE_TOKEN[cfg.SIDE_RIGHT] in up:
            if token != "RIGHT":
                continue
        elif SIDE_TOKEN[cfg.SIDE_LEFT] in up:
            if token != "LEFT":
                continue
        else:
            continue  # side not encoded (localizer/XR) -> skip
        for seq, rx in MATCHERS.items():
            if found[seq] is None and rx.search(sd):
                found[seq] = series_dir
                break
    return found


def _load_volume(series_dir: Path) -> np.ndarray:
    """Load a DICOM series dir as a 3D numpy volume (Z, H, W)."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        files = sorted(str(p) for p in series_dir.iterdir() if p.is_file())
    reader.SetFileNames(files)
    img = reader.Execute()
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _remove_path(path: Path) -> None:
    """Remove a file or directory if it exists."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _cleanup_tmp_dirs(root: Path) -> int:
    """Remove stale *.tmp directories left by interrupted runs."""
    removed = 0
    for d in root.iterdir():
        if d.is_dir() and d.name.endswith(".tmp"):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def _reset_derived_csvs() -> None:
    """Clear derived CSVs before a fresh extraction pass."""
    _remove_path(cfg.PROCESSED_INDEX)
    _remove_path(cfg.MISSING_CSV)


def _reset_processed_folders() -> int:
    """Remove all existing processed knee folders before a fresh rebuild."""
    removed = 0
    if not cfg.PROCESSED.exists():
        return removed
    for d in list(cfg.PROCESSED.iterdir()):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        elif d.name.endswith(".tmp"):
            d.unlink(missing_ok=True)
    return removed


def _central_indices(n: int, k: int, frac: float = 0.5) -> list:
    lo = int(round(n * (1 - frac) / 2))
    hi = int(round(n * (1 + frac) / 2)) - 1
    if hi <= lo:
        lo, hi = 0, max(n - 1, 0)
    if k > 1:
        return [int(round(lo + (hi - lo) * i / (k - 1))) for i in range(k)]
    return [(lo + hi) // 2]


def _save_slices(series_dir: Path, out_subdir: Path, seq_key: str) -> list:
    vol = _load_volume(series_dir)
    n = vol.shape[0]
    ids = _central_indices(n, cfg.N_SLICES, frac=getattr(cfg, 'FRAC', 0.5))
    flat = vol.reshape(-1)
    vmin = float(np.percentile(flat, cfg.CLIP_PERCENTILE[0]))
    vmax = float(np.percentile(flat, cfg.CLIP_PERCENTILE[1]))
    rng = (vmax - vmin) or 1.0
    out_subdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, zi in enumerate(ids):
        s = np.clip((vol[zi] - vmin) / rng, 0, 1)
        img = Image.fromarray((s * 255).astype(np.uint8))
        if img.size != (cfg.RESIZE, cfg.RESIZE):
            img = img.resize((cfg.RESIZE, cfg.RESIZE), Image.BILINEAR)
        p = out_subdir / f"{seq_key}_{i:02d}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def _extract_archive(ap: str, dest: Path) -> bool:
    """Extract one tar.gz archive into a complete cache directory.

    The extraction is guarded by a .complete marker inside the extracted
    directory, so interrupted runs never reuse a half-populated cache.
    """
    marker = dest / ".complete"
    if marker.exists():
        return True

    _remove_path(dest)
    tmp_dir = dest.parent / f"{dest.name}.tmp"
    _remove_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(ap, mode="r:gz") as tf:
            tf.extractall(tmp_dir)  # noqa: S202 (trusted local OAI data)
        _remove_path(dest)
        tmp_dir.rename(dest)
        marker.touch()
        return True
    except Exception:
        _remove_path(tmp_dir)
        return False


def _process_knee_atomic(targets: dict, final_dir: Path) -> dict | None:
    """Save all 9 PNGs into a temp dir, then atomically rename to final_dir.

    Returns a row dict mapping seq_00..seq_02 to final paths, or None on
    failure. The atomic rename guarantees final_dir only ever exists as a
    complete 9-file folder (or not at all), so a killed worker never leaves
    a partial folder that the index could reference.
    """
    tmp_dir = final_dir.parent / f"{final_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        for seq in cfg.SEQUENCES:
            _save_slices(targets[seq], tmp_dir, seq)
        # Verify all expected PNGs are present before committing.
        expected = len(cfg.SEQUENCES) * cfg.N_SLICES
        n_saved = sum(1 for f in tmp_dir.iterdir() if f.is_file())
        if n_saved != expected:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
        # Atomic rename (on same filesystem): final_dir appears complete.
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        tmp_dir.rename(final_dir)
        # Return final paths.
        row = {}
        for seq in cfg.SEQUENCES:
            for i in range(cfg.N_SLICES):
                row[f"{seq}_{i:02d}"] = str(final_dir / f"{seq}_{i:02d}.png")
        return row
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def process_participant(args) -> tuple:
    """Worker: process both knees of one participant. Returns (rows, missing).

    args = (pid, [(side, label), ...], archive_paths, keep_cache)
    """
    pid, knees, archive_paths, keep_cache = args
    rows, missing = [], []
    pid_dir = cfg.CACHE / pid
    if not keep_cache:
        _remove_path(pid_dir)
    pid_dir.mkdir(parents=True, exist_ok=True)

    # Extract every archive for this participant (once). The header/sequence
    # lives inside the tar, and gzip can't random-access, so we must extract
    # to disk. We delete it again at the end unless --keep-cache.
    for ap in archive_paths:
        stem = Path(ap).name.replace(".tar.gz", "")
        dest = pid_dir / stem
        _extract_archive(ap, dest)

    try:
        for side, label in knees:
            targets = _find_targets(pid_dir, int(side))
            missing_seqs = [k for k, v in targets.items() if v is None]
            if missing_seqs:
                missing.append({"ID": pid, "SIDE": int(side),
                                "reason": "missing:" + ",".join(missing_seqs)})
                continue
            out_subdir = cfg.PROCESSED / f"{pid}_{side}"
            try:
                row = {"ID": pid, "SIDE": int(side), "label": int(label)}
                final = _process_knee_atomic(targets, out_subdir)
                if final is None:
                    missing.append({"ID": pid, "SIDE": int(side),
                                    "reason": "proc:incomplete_slices"})
                    continue
                row.update(final)
                rows.append(row)
            except Exception as e:
                missing.append({"ID": pid, "SIDE": int(side),
                                "reason": f"proc:{type(e).__name__}:{e}"[:80]})
    finally:
        if not keep_cache:
            shutil.rmtree(pid_dir, ignore_errors=True)
    return rows, missing


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel workers (0 = all CPUs)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N participants (0 = all)")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep extracted DICOMs in cache/ (default: delete)")
    ap.add_argument("--reset-derived", action="store_true",
                    help="clear processed_index.csv and missing.csv before run")
    args = ap.parse_args()

    cfg.ensure_dirs()
    stale_tmp = _cleanup_tmp_dirs(cfg.PROCESSED)
    if stale_tmp:
        print(f"Cleaned {stale_tmp} stale processed .tmp dir(s)")
    workers = args.workers or (os.cpu_count() or 4)
    print("Loading labels + manifest ...")
    labels = pd.read_csv(cfg.LABELS_CSV, dtype={"ID": str, "SIDE": int})
    manifest = pd.read_csv(cfg.MRI_MANIFEST_CSV, dtype=str)
    dup_groups = int(labels.groupby(["ID", "SIDE"]).size().gt(1).sum())
    labels = dedupe_knee_rows(labels)
    if dup_groups:
        print(f"Collapsed {dup_groups} duplicate labeled knees")

    # Resumability: which knees are already done.
    done_keys = set()
    if args.reset_derived:
        print("Resetting derived CSVs for a fresh 3-slice extraction pass")
        _reset_derived_csvs()
        removed = _reset_processed_folders()
        if removed:
            print(f"Removed {removed} existing processed knee folder(s)")
    elif cfg.PROCESSED_INDEX.exists():
        d = pd.read_csv(cfg.PROCESSED_INDEX, dtype={"ID": str, "SIDE": int})
        if list(d.columns) != EXPECTED_COLUMNS:
            print("Existing processed_index.csv is for a different slice "
                  f"layout; resetting derived CSVs for the new {cfg.N_SLICES}-slice run")
            _reset_derived_csvs()
            removed = _reset_processed_folders()
            if removed:
                print(f"Removed {removed} existing processed knee folder(s)")
        else:
            done_keys = set(zip(d["ID"].astype(str), d["SIDE"].astype(int)))

    # Group knees by participant and drop participants with both knees done.
    archive_paths_by_id = manifest.groupby("ID")["archive_path"].apply(list)
    knees_by_id = (labels.groupby("ID")
                   .apply(lambda g: list(zip(g["SIDE"].astype(int),
                                             g["label"].astype(int)))))
    tasks = []
    skipped_participants = 0
    no_mri_missing = []
    for pid, knees in knees_by_id.items():
        if all((pid, s) in done_keys for s, _ in knees):
            skipped_participants += 1
            continue
        remaining = [(s, l) for (s, l) in knees if (pid, s) not in done_keys]
        if not remaining:
            skipped_participants += 1
            continue
        if pid not in archive_paths_by_id.index:
            for s, _ in remaining:
                no_mri_missing.append({
                    "ID": pid,
                    "SIDE": int(s),
                    "reason": f"no_mri:{cfg.NO_MRI_REASON}",
                })
            continue
        tasks.append((pid, remaining, list(archive_paths_by_id[pid]),
                      args.keep_cache))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Participants to process: {len(tasks)} "
          f"(skipped {skipped_participants} already done) | "
          f"workers: {workers}")

    all_rows, all_missing = [], []
    if no_mri_missing:
        _flush([], no_mri_missing)
        print(f"Marked {len(no_mri_missing)} knees with missing baseline MRI "
              f"reason: {cfg.NO_MRI_REASON}")
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_participant, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                rows, missing = fut.result()
            except Exception as e:
                all_missing.append({"ID": pid, "SIDE": -1,
                                    "reason": f"worker:{type(e).__name__}:{e}"[:80]})
                continue
            all_rows.extend(rows)
            all_missing.extend(missing)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done}/{len(tasks)} participants done | "
                      f"knees ok={len(all_rows)} missing={len(all_missing)} "
                      f"-> flushing")
                _flush(all_rows, all_missing)
                all_rows, all_missing = [], []

    _flush(all_rows, all_missing)
    print("\nDone. See data/processed_index.csv and logs/missing.csv")


def _flush(rows, missing) -> None:
    """Append new rows to the index/missing CSVs (header written once)."""
    if rows:
        df = pd.DataFrame(rows).reindex(columns=OUT_COLS)
        df.to_csv(cfg.PROCESSED_INDEX, mode="a",
                  header=not cfg.PROCESSED_INDEX.exists(), index=False)
    if missing:
        pd.DataFrame(missing).to_csv(cfg.MISSING_CSV, mode="a",
                                     header=not cfg.MISSING_CSV.exists(),
                                     index=False)


if __name__ == "__main__":
    main()
