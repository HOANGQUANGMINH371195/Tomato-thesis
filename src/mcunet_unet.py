"""MCUNet-inspired semantic segmentation model.

The encoder follows the Mobile Inverted Residual design used by MCUNet. The
decoder is a small U-Net-style decoder so the
classification-oriented MCUNet backbone can produce dense lesion masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class MobileInvertedResidual(nn.Module):
    """MobileNetV2/MCUNet inverted bottleneck with optional residual path."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int = 6) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden, 1, 1))
        layers.extend(
            [
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


class MCUNetEncoder(nn.Module):
    """MCUNet-style MobileNetV2 encoder with dense skip outputs."""

    def __init__(self, width_mult: float = 0.5) -> None:
        super().__init__()

        def width(value: int) -> int:
            return max(8, int(round(value * width_mult / 8)) * 8)

        # Compact MobileNetV2-style stage pattern used by the MCUNet encoder.
        first = width(32)
        stages = [
            (1, width(16), 1, 1),
            (6, width(24), 2, 2),
            (6, width(32), 3, 2),
            (6, width(64), 4, 2),
            (6, width(96), 3, 1),
            (6, width(160), 3, 2),
            (6, width(320), 1, 1),
        ]
        self.stem = ConvBNAct(3, first, 3, 2)
        self.stages = nn.ModuleList()
        in_channels = first
        for expand, out_channels, repeats, stride in stages:
            blocks = [MobileInvertedResidual(in_channels, out_channels, stride, expand)]
            blocks.extend(MobileInvertedResidual(out_channels, out_channels, 1, expand) for _ in range(repeats - 1))
            self.stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.out_channels = [width(16), width(24), width(32), width(64), width(320)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            # 112, 56, 28 and 14 resolution features, plus the 7 resolution bottleneck.
            if index in {0, 1, 2, 3}:
                skips.append(x)
        skips.append(x)
        return skips


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        target_size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MCUNetSegmentation(nn.Module):
    """Dense MCUNet-style segmentation model for 4-class lesion masks."""

    def __init__(self, num_classes: int = 4, width_mult: float = 0.5) -> None:
        super().__init__()
        self.encoder = MCUNetEncoder(width_mult)
        c112, c56, c28, c14, c7 = self.encoder.out_channels
        self.dec14 = DecoderBlock(c7, c14, 64)
        self.dec28 = DecoderBlock(64, c28, 48)
        self.dec56 = DecoderBlock(48, c56, 32)
        self.dec112 = DecoderBlock(32, c112, 24)
        self.dec224 = DecoderBlock(24, 0, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        x = skips[-1]
        x = self.dec14(x, skips[3])
        x = self.dec28(x, skips[2])
        x = self.dec56(x, skips[1])
        x = self.dec112(x, skips[0])
        x = self.dec224(x, None)
        return self.head(x)
