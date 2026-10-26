"""Build the 16-slice OAI image index and PNG cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg


N_SLICES_V2_CLEAN = 16
FRAC_V2_CLEAN = 0.80
PROCESSED_16S_V2 = cfg.DATA / "processed_16s_v2"
PROCESSED_INDEX_16S_V2 = cfg.DATA / "processed_index_16s_v2.csv"
MISSING_16S_V2 = cfg.LOGS / "missing_16s_v2.csv"


print("=" * 70)
print("16-SLICE V2 EXTRACTION")
print("=" * 70)
print(f"  N_SLICES   : {cfg.N_SLICES} -> {N_SLICES_V2_CLEAN}")
print(f"  FRAC       : {cfg.FRAC} -> {FRAC_V2_CLEAN}")
print(f"  OUTPUT DIR : {PROCESSED_16S_V2}")
print(f"  INDEX CSV  : {PROCESSED_INDEX_16S_V2}")
print(f"  MISSING CSV: {MISSING_16S_V2}")
print("=" * 70)

# Patch config before importing extract_preprocess so module-level OUT_COLS and
# EXPECTED_COLUMNS are computed for the v2 layout.
cfg.N_SLICES = N_SLICES_V2_CLEAN
cfg.FRAC = FRAC_V2_CLEAN
cfg.PROCESSED = PROCESSED_16S_V2
cfg.PROCESSED_INDEX = PROCESSED_INDEX_16S_V2
cfg.MISSING_CSV = MISSING_16S_V2
PROCESSED_16S_V2.mkdir(parents=True, exist_ok=True)

from extract_slices import main


if __name__ == "__main__":
    main()
