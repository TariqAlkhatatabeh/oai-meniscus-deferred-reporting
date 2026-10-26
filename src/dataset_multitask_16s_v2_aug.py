"""Dataset and split helpers for the 16-slice tri-plane model."""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config as cfg
from knee_utils import dedupe_knee_rows


N_SLICES = 16
PROCESSED_INDEX = cfg.DATA / "processed_index_16s_v2.csv"
MOAKS_COLS = ["MMTMA", "MMTMB", "MMTMP", "MMTLA", "MMTLB", "MMTLP"]
RECIPE_DESCRIPTIONS = {
    "baseline": (
        "Original 16s baseline: hflip p=0.5, rotation +/-10 deg p=0.5, "
        "gamma 0.8-1.2 p=0.5."
    ),
    "safe_mild": (
        "Anatomy-safe mild MRI recipe: no hflip, rotation +/-7 deg, "
        "translation <=4%, gamma 0.90-1.10, light noise/blur."
    ),
    "safe_strong": (
        "Anatomy-safe stronger MRI recipe: no hflip, rotation +/-10 deg, "
        "translation <=6%, gamma 0.85-1.15, contrast/noise/blur."
    ),
    "randaug_mri_lite": (
        "RandAugment-lite for grayscale MRI: sample two mild operations from "
        "rotate, translate, gamma, contrast, noise, blur; no hflip."
    ),
}


def _rotate_stack(stack: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """Rotate a `(slices, 1, H, W)` stack without importing torchvision."""
    return _affine_stack(stack, angle_degrees=angle_degrees)


def _affine_stack(stack: torch.Tensor,
                  angle_degrees: float = 0.0,
                  translate_frac: tuple[float, float] = (0.0, 0.0),
                  scale: float = 1.0) -> torch.Tensor:
    """Apply one affine transform to every slice in a stack."""
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    tx, ty = translate_frac
    theta = stack.new_tensor([[scale * cos_a, -scale * sin_a, tx],
                              [scale * sin_a, scale * cos_a, ty]])
    theta = theta.unsqueeze(0).expand(stack.size(0), -1, -1)
    grid = F.affine_grid(theta, stack.size(), align_corners=False)
    return F.grid_sample(
        stack, grid, mode="bilinear", padding_mode="zeros",
        align_corners=False)


def _blur_stack(stack: torch.Tensor, sigma: float) -> torch.Tensor:
    """Small Gaussian blur for acquisition-resolution robustness."""
    sigma = max(float(sigma), 0.1)
    radius = 1
    coords = torch.arange(-radius, radius + 1, dtype=stack.dtype,
                          device=stack.device)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, 3, 3)
    return F.conv2d(stack, kernel, padding=1)


def _clamp_dict(x):
    return {key: value.clamp(0.0, 1.0) for key, value in x.items()}


def describe_aug_recipe(recipe: str) -> str:
    if recipe not in RECIPE_DESCRIPTIONS:
        raise ValueError(f"Unknown augmentation recipe: {recipe}")
    return RECIPE_DESCRIPTIONS[recipe]


def _load_mt_index() -> pd.DataFrame:
    if not PROCESSED_INDEX.exists():
        raise FileNotFoundError(
            f"Missing 16-slice v2 index: {PROCESSED_INDEX}\n"
            "Run: python src/extract_16s_v2.py --workers 8"
        )
    idx = pd.read_csv(PROCESSED_INDEX, dtype={"ID": str})
    labels = pd.read_csv(cfg.LABELS_CSV, dtype={"ID": str})
    labels_sub = labels[["ID", "SIDE"] + MOAKS_COLS].copy()
    merged = idx.merge(labels_sub, on=["ID", "SIDE"], how="left",
                       suffixes=("", "_lbl"))

    for col in MOAKS_COLS:
        merged[col] = merged[col].fillna(0).astype(float)

    merged["_max_moaks"] = merged[MOAKS_COLS].max(axis=1)
    merged["severity"] = merged["_max_moaks"].apply(
        lambda value: 0 if value == 0 else (1 if value == 1 else 2))
    merged["severity"] = merged["severity"].astype(int)
    merged.drop(columns=["_max_moaks"], inplace=True)
    return merged


def _load_split_ids(split: str) -> set[str]:
    if cfg.SPLITS_CSV.exists():
        splits = pd.read_csv(cfg.SPLITS_CSV, dtype={"ID": str})
    else:
        from dataset import make_splits
        splits = make_splits()
    return set(splits.loc[splits["split"] == split, "ID"].astype(str))


def load_mt_split(split: str) -> pd.DataFrame:
    rows = _load_mt_index()
    ids = _load_split_ids(split)
    rows = rows[rows["ID"].astype(str).isin(ids)].reset_index(drop=True)
    rows = dedupe_knee_rows(rows)

    seq_cols = [
        f"{seq}_{idx:02d}"
        for seq in cfg.SEQUENCES
        for idx in range(N_SLICES)
    ]
    before = len(rows)
    keep = rows[seq_cols].apply(
        lambda row: all(Path(str(path)).exists() for path in row), axis=1)
    rows = rows[keep].reset_index(drop=True)
    dropped = before - len(rows)
    if dropped:
        print(f"  [warn] dropped {dropped} {split} knees with missing files")
    return rows


class MeniscusDatasetMT16SV2(Dataset):
    """16-slice multitask dataset returning `(x_dict, y, severity)`."""

    def __init__(self, rows: pd.DataFrame, augment: bool = False,
                 augment_recipe: str = "baseline"):
        if augment_recipe not in RECIPE_DESCRIPTIONS:
            raise ValueError(
                f"augment_recipe must be one of "
                f"{sorted(RECIPE_DESCRIPTIONS)}; got {augment_recipe!r}"
            )
        self.rows = rows.reset_index(drop=True)
        self.augment = augment
        self.augment_recipe = augment_recipe

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_stack(paths: list[str]) -> torch.Tensor:
        arrs = []
        for path in paths:
            img = Image.open(path).convert("L")
            arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
        vol = np.stack(arrs, axis=0)[:, None, :, :]
        return torch.from_numpy(vol)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        x = {}
        for seq in cfg.SEQUENCES:
            paths = [row[f"{seq}_{j:02d}"] for j in range(N_SLICES)]
            x[seq] = self._load_stack(paths)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        severity = torch.tensor(int(row["severity"]), dtype=torch.long)
        if self.augment:
            x = self._augment(x)
        return x, y, severity

    def _augment(self, x):
        if self.augment_recipe == "baseline":
            return self._augment_baseline(x)
        if self.augment_recipe == "safe_mild":
            return self._augment_safe_mild(x)
        if self.augment_recipe == "safe_strong":
            return self._augment_safe_strong(x)
        if self.augment_recipe == "randaug_mri_lite":
            return self._augment_randaug_mri_lite(x)
        raise ValueError(f"Unknown augmentation recipe: {self.augment_recipe}")

    def _augment_baseline(self, x):
        # Reproduces the original baseline exactly: hflip, rotation, gamma.
        if random.random() < 0.5:
            x = {key: torch.flip(value, dims=[3]) for key, value in x.items()}
        if random.random() < 0.5:
            angle = random.uniform(-10.0, 10.0)
            x = {key: _rotate_stack(value, angle) for key, value in x.items()}
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            x = {key: torch.clamp(value ** gamma, 0.0, 1.0)
                 for key, value in x.items()}
        return x

    def _augment_safe_mild(self, x):
        # No horizontal flip: medial/lateral anatomy and side are meaningful.
        if random.random() < 0.45:
            angle = random.uniform(-7.0, 7.0)
            x = {key: _rotate_stack(value, angle) for key, value in x.items()}
        if random.random() < 0.35:
            tx = random.uniform(-0.04, 0.04)
            ty = random.uniform(-0.04, 0.04)
            x = {key: _affine_stack(value, translate_frac=(tx, ty))
                 for key, value in x.items()}
        if random.random() < 0.45:
            gamma = random.uniform(0.90, 1.10)
            x = {key: value.clamp_min(0.0).pow(gamma)
                 for key, value in x.items()}
        if random.random() < 0.25:
            sigma = random.uniform(0.005, 0.015)
            x = {key: value + torch.randn_like(value) * sigma
                 for key, value in x.items()}
        if random.random() < 0.10:
            blur_sigma = random.uniform(0.5, 0.9)
            x = {key: _blur_stack(value, blur_sigma)
                 for key, value in x.items()}
        return _clamp_dict(x)

    def _augment_safe_strong(self, x):
        # A deliberately stronger but still grayscale-MRI-safe ablation.
        if random.random() < 0.60:
            angle = random.uniform(-10.0, 10.0)
            tx = random.uniform(-0.04, 0.04)
            ty = random.uniform(-0.04, 0.04)
            x = {key: _affine_stack(value, angle, (tx, ty))
                 for key, value in x.items()}
        if random.random() < 0.50:
            gamma = random.uniform(0.85, 1.15)
            x = {key: value.clamp_min(0.0).pow(gamma)
                 for key, value in x.items()}
        if random.random() < 0.35:
            factor = random.uniform(0.90, 1.10)
            x = {key: (value - value.mean()) * factor + value.mean()
                 for key, value in x.items()}
        if random.random() < 0.35:
            sigma = random.uniform(0.008, 0.025)
            x = {key: value + torch.randn_like(value) * sigma
                 for key, value in x.items()}
        if random.random() < 0.15:
            blur_sigma = random.uniform(0.5, 1.0)
            x = {key: _blur_stack(value, blur_sigma)
                 for key, value in x.items()}
        return _clamp_dict(x)

    def _apply_randaug_op(self, x, op_name):
        if op_name == "rotate":
            angle = random.uniform(-8.0, 8.0)
            return {key: _rotate_stack(value, angle)
                    for key, value in x.items()}
        if op_name == "translate":
            tx = random.uniform(-0.05, 0.05)
            ty = random.uniform(-0.05, 0.05)
            return {key: _affine_stack(value, translate_frac=(tx, ty))
                    for key, value in x.items()}
        if op_name == "gamma":
            gamma = random.uniform(0.88, 1.12)
            return {key: value.clamp_min(0.0).pow(gamma)
                    for key, value in x.items()}
        if op_name == "contrast":
            factor = random.uniform(0.88, 1.12)
            return {key: (value - value.mean()) * factor + value.mean()
                    for key, value in x.items()}
        if op_name == "noise":
            sigma = random.uniform(0.005, 0.020)
            return {key: value + torch.randn_like(value) * sigma
                    for key, value in x.items()}
        if op_name == "blur":
            blur_sigma = random.uniform(0.5, 0.9)
            return {key: _blur_stack(value, blur_sigma)
                    for key, value in x.items()}
        raise ValueError(f"Unknown randaug op: {op_name}")

    def _augment_randaug_mri_lite(self, x):
        ops = ["rotate", "translate", "gamma", "contrast", "noise", "blur"]
        for op_name in random.sample(ops, k=2):
            x = self._apply_randaug_op(x, op_name)
        return _clamp_dict(x)


def get_mt_loaders(batch_size: int = 4,
                   num_workers: int = 0,
                   pin_memory: bool = False,
                   aug_recipe: str = "baseline"):
    train_rows = load_mt_split("train")
    val_rows = load_mt_split("val")
    test_rows = load_mt_split("test")

    train_ds = MeniscusDatasetMT16SV2(
        train_rows, augment=True, augment_recipe=aug_recipe)
    val_ds = MeniscusDatasetMT16SV2(val_rows, augment=False)
    test_ds = MeniscusDatasetMT16SV2(test_rows, augment=False)

    y = train_rows["label"].astype(int).to_numpy()
    class_counts = torch.tensor(
        [(y == 0).sum(), (y == 1).sum()], dtype=torch.float32)
    weights = 1.0 / class_counts.clamp_min(1.0)
    sample_w = weights[torch.from_numpy(y.copy()).long()]
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(train_ds), replacement=True)

    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, **kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **kwargs)

    print(f"16-slice v2 multitask split sizes -> train:{len(train_ds)} "
          f"val:{len(val_ds)} test:{len(test_ds)}")
    print(f"Augmentation recipe: {aug_recipe} | "
          f"{describe_aug_recipe(aug_recipe)}")
    sev_counts = train_rows["severity"].value_counts().sort_index().to_dict()
    pos = int(train_rows["label"].sum())
    print(f"Train severity dist: {sev_counts}")
    print(f"Train tear/no-tear: {pos}/{len(train_ds) - pos}")
    return train_loader, val_loader, test_loader


def verify_no_leakage_mt() -> None:
    if cfg.SPLITS_CSV.exists():
        splits = pd.read_csv(cfg.SPLITS_CSV, dtype={"ID": str})
    else:
        from dataset import make_splits
        splits = make_splits()
    leaks = splits.groupby("ID")["split"].nunique().reset_index(name="n")
    bad = leaks[leaks["n"] > 1]
    assert bad.empty, f"DATA LEAK: {bad['ID'].head().tolist()}"
    print("No leakage: each participant is in exactly one split. "
          f"Split participant counts: {splits['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        rows = load_mt_split(split)
        print(f"{split}: {len(rows)} knees, {int(rows['label'].sum())} tears, "
              f"severity {rows['severity'].value_counts().sort_index().to_dict()}")
