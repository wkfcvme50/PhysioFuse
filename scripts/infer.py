#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiofuse.dataset import MaskedPairTransform
from physiofuse.evaluate import load_state_dict
from physiofuse.model import PhysioFuseNet


def preprocess_pair(
    eye_path: Path,
    nail_path: Path,
    eye_mask_path: Optional[Path],
    nail_mask_path: Optional[Path],
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = MaskedPairTransform(train=False, image_size=image_size)
    eye = Image.open(eye_path).convert("RGB")
    nail = Image.open(nail_path).convert("RGB")
    eye_mask = Image.open(eye_mask_path).convert("L") if eye_mask_path and eye_mask_path.exists() else Image.new("L", eye.size, 255)
    nail_mask = Image.open(nail_mask_path).convert("L") if nail_mask_path and nail_mask_path.exists() else Image.new("L", nail.size, 255)
    eye_tensor, _ = transform(eye, eye_mask)
    nail_tensor, _ = transform(nail, nail_mask)
    return eye_tensor.unsqueeze(0), nail_tensor.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhysioFuse inference for one paired sample.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "physiofuse.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eye", type=Path, required=True)
    parser.add_argument("--nail", type=Path, required=True)
    parser.add_argument("--eye-mask", type=Path)
    parser.add_argument("--nail-mask", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysioFuseNet(cfg).to(device)
    model.load_state_dict(load_state_dict(str(args.checkpoint), device))
    eye, nail = preprocess_pair(args.eye, args.nail, args.eye_mask, args.nail_mask, image_size=cfg["train"]["image_size"])
    model.eval()
    with torch.no_grad():
        pred = model(eye.to(device), nail.to(device))["hb"].item()
    print(f"{pred:.4f}")


if __name__ == "__main__":
    main()
