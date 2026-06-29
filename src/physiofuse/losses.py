from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def regression_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "smooth_l1") -> torch.Tensor:
    if mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if mode == "l1":
        return F.l1_loss(pred, target)
    if mode == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown regression loss: {mode}")


def consistency_loss(pe: torch.Tensor | None, pn: torch.Tensor | None, mode: str = "normalized_cosine") -> torch.Tensor:
    if pe is None or pn is None:
        raise ValueError("Consistency loss requires both eye and nail physiological embeddings.")
    pe_norm = F.normalize(pe, dim=1, eps=1e-8)
    pn_norm = F.normalize(pn, dim=1, eps=1e-8)
    if mode in {"normalized_cosine", "cosine"}:
        return (1.0 - F.cosine_similarity(pe_norm, pn_norm, dim=1)).mean()
    if mode in {"normalized_l2", "l2"}:
        return F.mse_loss(pe_norm, pn_norm)
    if mode in {"none", ""}:
        return pe.new_tensor(0.0)
    raise ValueError(f"Unknown consistency loss: {mode}")


def physiofuse_loss(
    out: dict[str, torch.Tensor | None],
    hb: torch.Tensor,
    eye: torch.Tensor,
    nail: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg.get("loss", cfg)
    model_cfg = cfg.get("model", cfg)

    main = regression_loss(out["hb"], hb, str(loss_cfg.get("regression_loss", "smooth_l1")))

    recon_terms = []
    if out.get("rec_eye") is not None:
        recon_terms.append(F.mse_loss(out["rec_eye"], eye))
    if out.get("rec_nail") is not None:
        recon_terms.append(F.mse_loss(out["rec_nail"], nail))
    recon = sum(recon_terms) / len(recon_terms) if recon_terms else hb.new_tensor(0.0)

    if model_cfg.get("input_mode", "both") == "both":
        consist = consistency_loss(out.get("pe"), out.get("pn"), str(loss_cfg.get("consistency_loss", "normalized_cosine")))
    else:
        consist = hb.new_tensor(0.0)

    total = (
        main
        + float(loss_cfg.get("lambda_reconstruction", 0.12)) * recon
        + float(loss_cfg.get("lambda_consistency", 0.03)) * consist
    )
    return total, {
        "total": float(total.detach().cpu().item()),
        "main": float(main.detach().cpu().item()),
        "reconstruction": float(recon.detach().cpu().item()),
        "consistency": float(consist.detach().cpu().item()),
    }

