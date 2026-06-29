#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiofuse.dataset import MaskedPairDataset, hb_bins, load_label_csv
from physiofuse.evaluate import evaluate_model, load_state_dict
from physiofuse.model import PhysioFuseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing PhysioFuse fold checkpoints.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "physiofuse.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_cv_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = load_label_csv(cfg["data"]["label_csv"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = StratifiedKFold(n_splits=int(cfg["train"]["cv_folds"]), shuffle=True, random_state=int(cfg["train"]["seed"]))
    bins = hb_bins(df["hb"].values)
    rows = []
    for fold, (_, val_idx) in enumerate(splitter.split(np.arange(len(df)), bins)):
        ckpt = args.checkpoint_dir / cfg["output"]["checkpoint_pattern"].format(fold=fold)
        model = PhysioFuseNet(cfg).to(device)
        model.load_state_dict(load_state_dict(str(ckpt), device))
        ds = MaskedPairDataset(cfg["data"]["data_dir"], df.iloc[val_idx].reset_index(drop=True), train=False, image_size=cfg["train"]["image_size"])
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])
        result = evaluate_model(model, loader, device)
        rows.append({"fold": fold, "checkpoint": str(ckpt), **result["metrics"]})
    summary = {"folds": rows}
    for key in ["mae", "rmse", "corr", "r2", "evs", "mean_physio_cosine"]:
        vals = [row[key] for row in rows if key in row]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

