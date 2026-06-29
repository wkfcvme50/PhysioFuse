from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import MaskedPairDataset, hb_bins, sample_weights
from .evaluate import evaluate_model
from .losses import physiofuse_loss
from .model import PhysioFuseNet, count_trainable_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-3) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.counter = 0

    def step(self, mae: float) -> bool:
        score = -float(mae)
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def make_loader(cfg: dict[str, Any], df: pd.DataFrame, train: bool) -> DataLoader:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    ds = MaskedPairDataset(
        data_cfg["data_dir"],
        df,
        train=train,
        image_size=int(train_cfg.get("image_size", 224)),
    )
    if train and train_cfg.get("weighted_sampler", True):
        sampler = WeightedRandomSampler(sample_weights(df["hb"].values), len(df), replacement=True)
        return DataLoader(
            ds,
            batch_size=int(train_cfg["batch_size"]),
            sampler=sampler,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=True,
        )
    return DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
    )


def train_one_fold(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold: int,
    out_dir: Path,
) -> dict[str, Any]:
    train_cfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    train_loader = make_loader(cfg, df_train, train=True)
    val_loader = make_loader(cfg, df_val, train=False)

    model = PhysioFuseNet(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["epochs"]),
        eta_min=float(train_cfg["min_lr"]),
    )
    stopper = EarlyStopping(int(train_cfg["patience"]))
    best_mae = float("inf")
    ckpt_name = cfg["output"].get("checkpoint_pattern", "physiofuse_best_fold{fold}.pth").format(fold=fold)
    best_path = fold_dir / ckpt_name
    history: list[dict[str, float]] = []

    for epoch in range(int(train_cfg["epochs"])):
        if epoch < int(train_cfg["warmup_epochs"]):
            lr = float(train_cfg["lr"]) * (epoch + 1) / float(train_cfg["warmup_epochs"])
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            scheduler.step()

        model.train()
        loss_sums = {"total": 0.0, "main": 0.0, "reconstruction": 0.0, "consistency": 0.0}
        for batch in train_loader:
            eye = batch["eye"].to(device)
            nail = batch["nail"].to(device)
            hb = batch["hb"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(eye, nail)
            loss, parts = physiofuse_loss(out, hb, eye, nail, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in loss_sums:
                loss_sums[key] += float(parts[key])

        denom = max(1, len(train_loader))
        val = evaluate_model(model, val_loader, device)["metrics"]
        row = {
            "epoch": float(epoch + 1),
            "train_loss": loss_sums["total"] / denom,
            "train_main": loss_sums["main"] / denom,
            "train_reconstruction": loss_sums["reconstruction"] / denom,
            "train_consistency": loss_sums["consistency"] / denom,
            **{f"val_{k}": float(v) for k, v in val.items() if isinstance(v, (int, float))},
        }
        history.append(row)
        print(
            f"[fold {fold}] epoch {epoch + 1:03d}/{train_cfg['epochs']} "
            f"loss={row['train_loss']:.4f} val_mae={val['mae']:.4f} val_corr={val['corr']:.4f}"
        )

        if val["mae"] < best_mae:
            best_mae = float(val["mae"])
            torch.save(model.state_dict(), best_path)
        if stopper.step(float(val["mae"])):
            break

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    final = evaluate_model(model, val_loader, device)
    pred_path = fold_dir / "predictions.npz"
    np.savez(
        pred_path,
        predictions=final["predictions"],
        ground_truth=final["targets"],
        names=final["names"],
    )
    with (fold_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {
        "fold": int(fold),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "checkpoint": str(best_path),
        "predictions": str(pred_path),
        "model_params": count_trainable_params(model),
        **final["metrics"],
    }


def run_cv(cfg: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    set_seed(int(cfg["train"]["seed"]))
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = int(cfg["train"]["cv_folds"])
    bins = hb_bins(df["hb"].values)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg["train"]["seed"]))

    rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(df)), bins)):
        rows.append(train_one_fold(cfg, df, train_idx, val_idx, fold, out_dir))

    summary: dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(df)),
        "n_folds": folds,
        "folds": rows,
    }
    for key in ["mae", "rmse", "corr", "r2", "evs", "mean_physio_cosine"]:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))
    with (out_dir / "cv_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

