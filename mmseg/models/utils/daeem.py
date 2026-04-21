import torch
import torch.nn as nn


class DAEEM(nn.Module):
    """Direction-Aware Edge Enhancement Module.

    The module follows a three-step design:
    1. Multi-orientation perception with horizontal, vertical and local
       context branches.
    2. Interference suppression via an adaptive spatial reliability map.
    3. Residual fusion back to the shallow feature stream.
    """

    def __init__(self, in_channels, kernel_size=7, dilation=2, reduction=4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd for symmetric padding.')

        hidden_channels = max(in_channels // reduction, 4)
        pad = kernel_size // 2

        self.horizontal_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.vertical_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.orientation_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.suppression = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.residual_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        horizontal = self.horizontal_branch(x)
        vertical = self.vertical_branch(x)
        context = self.context_branch(x)

        oriented = self.orientation_fusion(
            torch.cat([horizontal, vertical, context], dim=1))
        suppression_map = self.suppression(oriented)
        refined = oriented * suppression_map

        fused = self.residual_fusion(torch.cat([x, refined], dim=1))
        out = x + self.gamma * fused
        return out, suppression_map
