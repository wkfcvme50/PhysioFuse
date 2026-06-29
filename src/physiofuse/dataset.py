from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_label_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"name", "hb"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    out = df.loc[:, ["name", "hb"]].copy()
    out["name"] = out["name"].astype(str)
    out["hb"] = out["hb"].astype(float)
    return out.reset_index(drop=True)


def hb_bins(values: np.ndarray | list[float]) -> np.ndarray:
    hb = np.asarray(values, dtype=np.float32)
    bins = np.zeros(len(hb), dtype=np.int64)
    bins[(hb >= 9.0) & (hb < 11.0)] = 1
    bins[(hb >= 11.0) & (hb < 13.0)] = 2
    bins[hb >= 13.0] = 3
    return bins


def sample_weights(values: np.ndarray | list[float]) -> torch.Tensor:
    hb = np.asarray(values, dtype=np.float32)
    weights = np.ones(len(hb), dtype=np.float32)
    weights[hb < 9.0] = 4.0
    weights[(hb >= 9.0) & (hb < 11.0)] = 2.5
    weights[(hb >= 11.0) & (hb < 13.0)] = 1.0
    weights[hb >= 13.0] = 1.5
    return torch.as_tensor(weights, dtype=torch.double)


def image_path_for(data_dir: str | Path, modality: str, name: str) -> Path:
    data_dir = Path(data_dir)
    if modality == "eye":
        return data_dir / "conjunctiva" / f"{name}.jpg"
    if modality == "nail":
        return data_dir / "nail" / f"{name}.jpg"
    raise ValueError(f"Unknown modality: {modality}")


def mask_path_for(data_dir: str | Path, modality: str, name: str) -> Path:
    data_dir = Path(data_dir)
    if modality == "eye":
        return data_dir / "conjunctiva" / "conjunctiva_masks" / f"{name}.png"
    if modality == "nail":
        return data_dir / "nail" / "nail_masks" / f"{name}.png"
    raise ValueError(f"Unknown modality: {modality}")


def pad_to_square(img: Image.Image, fill: int | tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    out = Image.new(img.mode, (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def load_mask(mask_path: Path, image_size: tuple[int, int]) -> Image.Image:
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        return mask
    return Image.new("L", image_size, 255)


class MaskedPairTransform:
    def __init__(self, train: bool, image_size: int = 224) -> None:
        self.train = bool(train)
        self.image_size = int(image_size)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = pad_to_square(image, (0, 0, 0))
        mask = pad_to_square(mask, 0)
        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self.train:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = float(torch.empty(1).uniform_(-6.0, 6.0).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            max_shift = int(round(self.image_size * 0.02))
            translate = (
                int(torch.randint(-max_shift, max_shift + 1, (1,)).item()),
                int(torch.randint(-max_shift, max_shift + 1, (1,)).item()),
            )
            scale = float(torch.empty(1).uniform_(0.98, 1.02).item())
            image = TF.affine(image, 0.0, translate, scale, [0.0, 0.0], InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, 0.0, translate, scale, [0.0, 0.0], InterpolationMode.NEAREST, fill=0)

        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        image_tensor = TF.to_tensor(image) * mask_tensor
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return image_tensor, mask_tensor


class MaskedPairDataset(Dataset):
    def __init__(self, data_dir: str | Path, df: pd.DataFrame, train: bool, image_size: int = 224) -> None:
        self.data_dir = Path(data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = MaskedPairTransform(train=train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.df)

    def _load_modality(self, name: str, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = image_path_for(self.data_dir, modality, name)
        image = Image.open(image_path).convert("RGB")
        mask = load_mask(mask_path_for(self.data_dir, modality, name), image.size)
        return self.transform(image, mask)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        name = str(row["name"])
        eye, eye_mask = self._load_modality(name, "eye")
        nail, nail_mask = self._load_modality(name, "nail")
        return {
            "eye": eye,
            "nail": nail,
            "hb": torch.tensor(float(row["hb"]), dtype=torch.float32),
            "name": name,
            "eye_mask_area": eye_mask.mean().float(),
            "nail_mask_area": nail_mask.mean().float(),
        }


def validate_dataset_layout(data_dir: str | Path, df: pd.DataFrame, limit: int = 20) -> list[str]:
    missing: list[str] = []
    for name in df["name"].astype(str):
        for modality in ("eye", "nail"):
            path = image_path_for(data_dir, modality, name)
            if not path.exists():
                missing.append(str(path))
                if len(missing) >= limit:
                    return missing
    return missing
