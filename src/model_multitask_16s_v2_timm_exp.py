"""Tri-plane ConvNeXt model for tear and severity prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


def _install_torchvision_stub_for_timm() -> None:
    """Avoid loading broken native torchvision image ops during timm import.

    Timm imports a few optional torchvision helpers at module import time. The
    current Windows environment can access-violate while loading torchvision's
    native image extension, even though this project does not use torchvision
    datasets, transforms, image IO, or FX extraction. A tiny stub for the two
    optional symbols timm asks for is enough for ConvNeXt/ViT creation.
    """
    if "torchvision" in sys.modules:
        return
    import types
    from enum import Enum

    tv = types.ModuleType("torchvision")
    tv.__path__ = []
    tv.__file__ = __file__

    ops = types.ModuleType("torchvision.ops")
    ops.__file__ = __file__
    misc = types.ModuleType("torchvision.ops.misc")
    misc.__file__ = __file__

    class FrozenBatchNorm2d(nn.BatchNorm2d):
        pass

    misc.FrozenBatchNorm2d = FrozenBatchNorm2d
    ops.misc = misc

    models = types.ModuleType("torchvision.models")
    models.__file__ = __file__
    feature_extraction = types.ModuleType("torchvision.models.feature_extraction")
    feature_extraction.__file__ = __file__

    def _feature_extraction_unavailable(*_args, **_kwargs):
        raise ImportError("torchvision feature extraction is disabled here")

    feature_extraction.create_feature_extractor = _feature_extraction_unavailable
    feature_extraction.get_graph_node_names = _feature_extraction_unavailable
    models.feature_extraction = feature_extraction

    datasets = types.ModuleType("torchvision.datasets")
    datasets.__file__ = __file__

    class _UnavailableDataset:
        def __init__(self, *_args, **_kwargs):
            raise ImportError("torchvision datasets are disabled here")

    for name in (
        "CIFAR100", "CIFAR10", "MNIST", "KMNIST", "FashionMNIST",
        "ImageFolder", "Places365", "INaturalist", "QMNIST", "ImageNet",
    ):
        setattr(datasets, name, _UnavailableDataset)

    transforms = types.ModuleType("torchvision.transforms")
    transforms.__file__ = __file__
    transforms_functional = types.ModuleType("torchvision.transforms.functional")
    transforms_functional.__file__ = __file__

    class _InterpolationMode(Enum):
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1

    class _IdentityTransform:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, x):
            return x

    def _identity_image(x, *_args, **_kwargs):
        return x

    def _pil_to_tensor(pic):
        import numpy as np
        import torch

        arr = np.asarray(pic)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = np.moveaxis(arr, -1, 0)
        return torch.from_numpy(arr.copy())

    def _to_tensor(pic):
        tensor = _pil_to_tensor(pic).float()
        return tensor / 255.0 if tensor.numel() and tensor.max() > 1 else tensor

    transforms.ToTensor = _IdentityTransform
    transforms.Compose = lambda items: _IdentityTransform()
    transforms_functional.InterpolationMode = _InterpolationMode
    transforms_functional.pil_to_tensor = _pil_to_tensor
    transforms_functional.to_tensor = _to_tensor
    for name in (
        "resize", "resized_crop", "crop", "pad", "center_crop", "hflip",
        "vflip", "rotate", "affine", "adjust_brightness", "adjust_contrast",
        "adjust_saturation", "adjust_hue", "adjust_gamma", "normalize",
    ):
        setattr(transforms_functional, name, _identity_image)

    def _transforms_getattr(_name):
        if _name.startswith("__"):
            raise AttributeError(_name)
        return _IdentityTransform

    def _functional_getattr(_name):
        if _name.startswith("__"):
            raise AttributeError(_name)
        return _identity_image

    transforms.__getattr__ = _transforms_getattr
    transforms.functional = transforms_functional
    transforms_functional.__getattr__ = _functional_getattr

    tv.ops = ops
    tv.models = models
    tv.datasets = datasets
    tv.transforms = transforms
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.ops.misc"] = misc
    sys.modules["torchvision.models"] = models
    sys.modules["torchvision.models.feature_extraction"] = feature_extraction
    sys.modules["torchvision.datasets"] = datasets
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = transforms_functional


_install_torchvision_stub_for_timm()
import timm


CONVNEXT_BASE_WEIGHTS = cfg.PRETRAINED_CONVNEXT
VIT_BASE_AUGREG_WEIGHTS = cfg.PRETRAINED_VIT


def expected_spatial_size(backbone_name: str) -> int:
    try:
        meta = timm.get_pretrained_cfg_value(backbone_name, "input_size")
        if meta:
            return int(meta[-1])
    except Exception:
        pass
    return 224


def _load_safetensors(path: Path) -> dict:
    from safetensors.torch import load_file

    if not path.exists():
        raise FileNotFoundError(path)
    return load_file(str(path))


def _adapt_state_for_single_channel(model: nn.Module, state: dict) -> dict:
    own = model.state_dict()
    adapted = {}
    for key, value in state.items():
        if key not in own:
            continue
        target = own[key]
        if target.shape == value.shape:
            adapted[key] = value
            continue
        if (
            value.ndim == 4
            and target.ndim == 4
            and value.shape[1] == 3
            and target.shape[1] == 1
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
        ):
            adapted[key] = value.mean(dim=1, keepdim=True)
    return adapted


def make_local_timm_backbone(backbone_name: str, weights_path: Path,
                             grad_checkpointing: bool = False) -> nn.Module:
    net = timm.create_model(
        backbone_name,
        pretrained=False,
        in_chans=1,
        num_classes=0,
        global_pool="avg",
    )
    state = _load_safetensors(weights_path)
    adapted = _adapt_state_for_single_channel(net, state)
    missing, unexpected = net.load_state_dict(adapted, strict=False)
    real_unexpected = [
        key for key in unexpected
        if not key.startswith("head.") and not key.startswith("fc.")
    ]
    if real_unexpected:
        print(f"  [warn] unexpected backbone keys: {real_unexpected[:6]} ...")
    if grad_checkpointing and hasattr(net, "set_grad_checkpointing"):
        net.set_grad_checkpointing(True)
        print(f"  enabled gradient checkpointing for {backbone_name}")
    print(
        f"  loaded local {backbone_name} weights -> {weights_path.name} "
        f"({len(adapted)} tensors, missing={len(missing)})"
    )
    return net


class PlaneMeanBranchTimm(nn.Module):
    """One plane: timm 1-channel slice encoder + mean pooling over slices."""

    def __init__(self, backbone_name: str, weights_path: Path,
                 grad_checkpointing: bool = False):
        super().__init__()
        self.net = make_local_timm_backbone(
            backbone_name, weights_path, grad_checkpointing)
        self.feat_dim = int(self.net.num_features)
        self.spatial = expected_spatial_size(backbone_name)

    def forward(self, x):
        batch, n_slices, channels, height, width = x.shape
        x = x.reshape(batch * n_slices, channels, height, width)
        if (height, width) != (self.spatial, self.spatial):
            x = F.interpolate(
                x, size=(self.spatial, self.spatial),
                mode="bilinear", align_corners=False)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        feats = self.net(x)
        return feats.reshape(batch, n_slices, -1).mean(dim=1)


class TriPlaneMultiTask16SV2Timm(nn.Module):
    """Tri-plane mean-pooled multitask model with local timm backbones."""

    def __init__(self, backbone_name: str, weights_path: Path,
                 dropout: float = 0.3, n_severity: int = 3,
                 grad_checkpointing: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.weights_path = str(weights_path)
        self.planes = list(cfg.SEQUENCES)
        self.branches = nn.ModuleDict({
            plane: PlaneMeanBranchTimm(
                backbone_name, weights_path, grad_checkpointing)
            for plane in self.planes
        })

        feat_dim = self.branches[self.planes[0]].feat_dim
        cat_dim = feat_dim * len(self.planes)

        self.main_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_severity),
        )

    def encoder_parameters(self):
        return self.branches.parameters()

    def head_parameters(self):
        for module in (self.main_head, self.aux_head):
            yield from module.parameters()

    def set_encoder_trainable(self, trainable: bool) -> None:
        for param in self.branches.parameters():
            param.requires_grad = trainable

    def forward(self, x_dict):
        feats = [self.branches[plane](x_dict[plane]) for plane in self.planes]
        cat = torch.cat(feats, dim=1)
        return self.main_head(cat), self.aux_head(cat)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TriPlaneMultiTask16SV2Timm(
        "convnext_base.fb_in22k_ft_in1k",
        CONVNEXT_BASE_WEIGHTS,
        grad_checkpointing=False,
    ).to(device)
    x = {
        plane: torch.randn(1, 16, 1, cfg.RESIZE, cfg.RESIZE, device=device)
        for plane in cfg.SEQUENCES
    }
    with torch.no_grad():
        main, aux = model(x)
    print(f"main: {tuple(main.shape)} aux: {tuple(aux.shape)}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
