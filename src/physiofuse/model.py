from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_dim: int, channels: int = 256) -> None:
        super().__init__()
        self.channels = int(channels)
        self.fc = nn.Linear(in_dim, channels * 7 * 7)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels // 2, channels // 4, 3, 1, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels // 4, channels // 8, 3, 1, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels // 8, channels // 16, 3, 1, 1),
            nn.BatchNorm2d(channels // 16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels // 16, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z).view(-1, self.channels, 7, 7)
        return self.net(z)


class HighFreqConfound(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.blur = nn.AvgPool2d(5, 1, 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x - self.blur(x))


class PhysioFuseNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        feature_dim = int(model_cfg.get("feature_dim", 384))
        physio_dim = int(model_cfg.get("physio_dim", 64))
        confound_dim = int(model_cfg.get("confound_dim", 128))
        dropout = float(model_cfg.get("dropout", 0.25))
        pretrained = bool(model_cfg.get("pretrained", False))

        self.input_mode = str(model_cfg.get("input_mode", "both"))
        self.has_eye = self.input_mode in ("both", "eye_only")
        self.has_nail = self.input_mode in ("both", "nail_only")
        self.use_high_freq_confound = bool(model_cfg.get("use_high_freq_confound", True))
        self.use_reconstruction_loss = bool(model_cfg.get("use_reconstruction_loss", True))

        if self.has_eye:
            self.eye_backbone = timm.create_model(model_cfg["eye_backbone"], pretrained=pretrained, num_classes=0)
            self.eye_project = nn.Sequential(
                nn.Linear(self.eye_backbone.num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, feature_dim),
            )
            self.eye_physio = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, physio_dim),
            )
            self.eye_confound = HighFreqConfound(confound_dim) if self.use_high_freq_confound else None
            rec_dim = physio_dim + confound_dim if self.use_high_freq_confound else physio_dim
            self.eye_decoder = Decoder(rec_dim) if self.use_reconstruction_loss else None
        else:
            self.eye_backbone = self.eye_project = self.eye_physio = self.eye_confound = self.eye_decoder = None

        if self.has_nail:
            self.nail_backbone = timm.create_model(model_cfg["nail_backbone"], pretrained=pretrained, num_classes=0)
            self.nail_project = nn.Sequential(
                nn.Linear(self.nail_backbone.num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, feature_dim),
            )
            self.nail_physio = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, physio_dim),
            )
            self.nail_confound = HighFreqConfound(confound_dim) if self.use_high_freq_confound else None
            rec_dim = physio_dim + confound_dim if self.use_high_freq_confound else physio_dim
            self.nail_decoder = Decoder(rec_dim) if self.use_reconstruction_loss else None
        else:
            self.nail_backbone = self.nail_project = self.nail_physio = self.nail_confound = self.nail_decoder = None

        fusion_in = physio_dim * 2 if self.input_mode == "both" else physio_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _encode_eye(self, eye: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.has_eye:
            return None, None, None
        feature = self.eye_project(self.eye_backbone(eye))
        physio = self.eye_physio(feature)
        confound = self.eye_confound(eye) if self.eye_confound is not None else None
        rec = None
        if self.eye_decoder is not None:
            rec_input = torch.cat([physio, confound], dim=1) if confound is not None else physio
            rec = self.eye_decoder(rec_input)
        return physio, confound, rec

    def _encode_nail(self, nail: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.has_nail:
            return None, None, None
        feature = self.nail_project(self.nail_backbone(nail))
        physio = self.nail_physio(feature)
        confound = self.nail_confound(nail) if self.nail_confound is not None else None
        rec = None
        if self.nail_decoder is not None:
            rec_input = torch.cat([physio, confound], dim=1) if confound is not None else physio
            rec = self.nail_decoder(rec_input)
        return physio, confound, rec

    def forward(self, eye: torch.Tensor, nail: torch.Tensor) -> dict[str, torch.Tensor | None]:
        pe, ce, rec_eye = self._encode_eye(eye)
        pn, cn, rec_nail = self._encode_nail(nail)

        if self.input_mode == "both":
            fusion = torch.cat([pe, pn], dim=1)
        elif self.input_mode == "eye_only":
            fusion = pe
        elif self.input_mode == "nail_only":
            fusion = pn
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        pred = self.regressor(fusion).squeeze(-1)
        return {
            "hb": pred,
            "pe": pe,
            "pn": pn,
            "ce": ce,
            "cn": cn,
            "rec_eye": rec_eye,
            "rec_nail": rec_nail,
        }


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

