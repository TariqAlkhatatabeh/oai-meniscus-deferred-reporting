"""Shared paths and constants.

Override the root paths with environment variables when the data are not in
the default repository-relative locations.
"""
from pathlib import Path
import os

PROJECT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MENISCUS_DATA", PROJECT / "data"))
RESULTS = Path(os.environ.get("MENISCUS_RESULTS", PROJECT / "results"))
WEIGHTS = Path(os.environ.get("MENISCUS_WEIGHTS", PROJECT / "weights"))
MODELS = Path(os.environ.get("MENISCUS_MODELS", PROJECT / "models"))
LOGS = Path(os.environ.get("MENISCUS_LOGS", PROJECT / "logs"))
CACHE = Path(os.environ.get("MENISCUS_CACHE", PROJECT / "cache"))

# Source data; these are not distributed with the repository.
OAI_MOAKS = Path(os.environ.get("OAI_MOAKS", DATA / "kMRI_SQ_MOAKS_BICL00.txt"))
OAI_MRI_ROOT = Path(os.environ.get("OAI_MRI_ROOT", DATA / "OAI_MRI"))
MRNET_ROOT = Path(os.environ.get("MRNET_ROOT", DATA / "MRNet-v1.0"))

# Model assets; these are not distributed with the repository.
PRETRAINED_CONVNEXT = Path(
    os.environ.get(
        "PRETRAINED_CONVNEXT",
        MODELS / "convnext_base.fb_in22k_ft_in1k.safetensors",
    )
)
PRETRAINED_VIT = Path(
    os.environ.get(
        "PRETRAINED_VIT",
        MODELS / "vit_base_patch16_224.augreg_in21k.safetensors",
    )
)
MODEL_QWEN_TEXT = Path(os.environ.get("QWEN_TEXT_MODEL", MODELS / "Qwen3.5-9B"))
MODEL_QWEN_VL = Path(
    os.environ.get("QWEN_VL_MODEL", MODELS / "Qwen2.5-VL-7B-Instruct")
)

# Image layout.
SEQUENCES = ["dess", "cor_mpr", "ax_mpr"]
SIDE_RIGHT = 1
SIDE_LEFT = 2
N_SLICES = 16
N_SLICES_V2 = 16
FRAC = 0.80
RESIZE = 256
CLIP_PERCENTILE = (0.5, 99.5)
PROCESSED = DATA / "processed_16s_v2"
PROCESSED_INDEX = DATA / "processed_index_16s_v2.csv"
MISSING_CSV = LOGS / "missing_16s_v2.csv"
LABELS_CSV = DATA / "labels.csv"
MRI_MANIFEST_CSV = DATA / "mri_manifest.csv"
SERIES_CATALOG = DATA / "series_catalog.csv"
SPLITS_CSV = DATA / "splits.csv"
MOAKS_TXT = OAI_MOAKS
MRI_ROOT = OAI_MRI_ROOT

TEAR_COLS = [
    "V00MMTMA", "V00MMTMB", "V00MMTMP",
    "V00MMTLA", "V00MMTLB", "V00MMTLP",
]
RENAME_COLS = {c: c.replace("V00MMT", "MMT") for c in TEAR_COLS}
NO_MRI_REASON = "baseline MRI is not available"
SEQ_MATCHERS = {
    "dess": r"SAG.*3D.*DESS",
    "cor_mpr": r"COR.*MPR",
    "ax_mpr": r"AX.*MPR",
}

BACKBONE = "convnext_base"
WEIGHT_DECAY = 1e-4
SEED = 42


def ensure_dirs():
    for path in (DATA, RESULTS, WEIGHTS, MODELS, LOGS, CACHE, DATA / "processed_16s_v2"):
        path.mkdir(parents=True, exist_ok=True)
