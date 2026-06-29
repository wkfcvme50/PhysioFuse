from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    corr = 0.0
    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    var_y = float(np.var(y_true))
    evs = 0.0 if var_y <= 1e-12 else float(1.0 - np.var(err) / var_y)
    return {"mae": mae, "rmse": rmse, "corr": corr, "r2": r2, "evs": evs}


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    physio_cos: list[np.ndarray] = []

    for batch in loader:
        eye = batch["eye"].to(device)
        nail = batch["nail"].to(device)
        hb = batch["hb"].to(device)
        out = model(eye, nail)
        pred = out["hb"].detach().cpu().numpy()
        preds.append(pred)
        targets.append(hb.detach().cpu().numpy())
        names.extend([str(x) for x in batch.get("name", [])])

        pe = out.get("pe")
        pn = out.get("pn")
        if pe is not None and pn is not None:
            cos = torch.nn.functional.cosine_similarity(
                torch.nn.functional.normalize(pe, dim=1),
                torch.nn.functional.normalize(pn, dim=1),
                dim=1,
            )
            physio_cos.append(cos.detach().cpu().numpy())

    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    y_true = np.concatenate(targets) if targets else np.array([], dtype=np.float32)
    metrics = regression_metrics(y_true, y_pred)
    metrics["n"] = int(len(y_true))
    if physio_cos:
        metrics["mean_physio_cosine"] = float(np.mean(np.concatenate(physio_cos)))
    return {"metrics": metrics, "predictions": y_pred, "targets": y_true, "names": np.asarray(names, dtype=object)}


def load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

