import torch
import torch.nn as nn


def _make_group_norm(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DAEEM(nn.Module):
    """Shallow boundary enhancement branch for decoder-side fusion."""

    def __init__(self, in_channels, kernel_size=7, dilation=2, reduction=4,
                 max_groups=8):
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
            _make_group_norm(in_channels, max_groups=max_groups),
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
            _make_group_norm(in_channels, max_groups=max_groups),
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
            _make_group_norm(in_channels, max_groups=max_groups),
            nn.ReLU(inplace=True),
        )

        self.branch_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, hidden_channels, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self.orientation_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            _make_group_norm(in_channels, max_groups=max_groups),
            nn.ReLU(inplace=True),
        )

        self.confidence = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,
                      bias=False),
            _make_group_norm(hidden_channels, max_groups=max_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            _make_group_norm(in_channels, max_groups=max_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      bias=False),
            _make_group_norm(in_channels, max_groups=max_groups),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        horizontal = self.horizontal_branch(x)
        vertical = self.vertical_branch(x)
        context = self.context_branch(x)

        stacked = torch.cat([horizontal, vertical, context], dim=1)
        selector = self.branch_selector(stacked)
        weighted = torch.cat([
            horizontal * selector[:, 0:1],
            vertical * selector[:, 1:2],
            context * selector[:, 2:3],
        ], dim=1)

        oriented = self.orientation_fusion(weighted)
        boundary_logits = self.confidence(oriented)
        confidence = torch.sigmoid(boundary_logits)
        refined = self.refine(torch.cat([x, oriented * confidence], dim=1))
        enhanced = x + self.gamma * refined
        return enhanced, boundary_logits
