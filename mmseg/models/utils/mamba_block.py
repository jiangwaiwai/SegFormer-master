import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_SSM = True
except ImportError:
    HAS_CUDA_SSM = False


def selective_scan_pure_pytorch(u, delta, A, B, C, D=None):
    """Pure PyTorch implementation of Selective State Space Model (S6).

    Args:
        u (torch.Tensor): Input sequence of shape (B, L, D_in).
        delta (torch.Tensor): Discretization step of shape (B, L, D_in), after softplus.
        A (torch.Tensor): Continuous state decay matrix of shape (D_in, N), negative.
        B (torch.Tensor): Input matrix of shape (B, L, N).
        C (torch.Tensor): Output matrix of shape (B, L, N).
        D (torch.Tensor, optional): Direct feedthrough skip of shape (D_in,).

    Returns:
        torch.Tensor: Output sequence of shape (B, L, D_in).
    """
    batch, L, d_inner = u.shape
    d_state = A.shape[1]

    # Discretize continuous parameters A and B:
    # delta: (B, L, D_in, 1), A: (1, 1, D_in, N) -> dA: (B, L, D_in, N)
    dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_inner, d_state))
    # delta: (B, L, D_in, 1), B: (B, L, 1, N) -> dB: (B, L, D_in, N)
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)
    # dBu: (B, L, D_in, N)
    dBu = dB * u.unsqueeze(-1)

    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dBu[:, t]
        # Contract over state dimension N with C[:, t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + u * D.view(1, 1, -1)
    return y


class MambaBlock(nn.Module):
    """Visual Mamba Block for sequence and 2D spatial feature modelling.

    Strictly implements the Selective State Space Model (S6) with visual
    bidirectional scanning and local depthwise convolution. Supports both
    accelerated CUDA selective scan (if mamba_ssm is installed on the server)
    and pure PyTorch fallback (zero external dependencies required).

    Args:
        dim (int): Feature channel dimension.
        d_state (int): SSM latent state expansion. Default: 16.
        d_conv (int): Local convolution kernel size. Default: 4.
        expand (int): Channel expansion factor. Default: 2.
        dt_rank (int, optional): Rank of dt projection. Defaults to ceil(dim / 16).
        dt_min (float): Minimum dt initialization value. Default: 0.001.
        dt_max (float): Maximum dt initialization value. Default: 0.1.
        dt_init_floor (float): Minimum floor for dt initialization. Default: 1e-4.
        drop_path (float): DropPath rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        bimamba (bool): Whether to perform bidirectional scanning. Default: True.
    """

    def __init__(self,
                 dim,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank=None,
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init_floor=1e-4,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 bimamba=True):
        super(MambaBlock, self).__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)
        self.dt_rank = dt_rank or math.ceil(self.dim / 16)
        self.bimamba = bimamba

        # Pre-normalization
        self.norm = norm_layer(dim)

        # In-projection to SSM branch (u) and gate branch (z)
        self.in_proj = nn.Linear(self.dim, 2 * self.d_inner, bias=False)

        # Local spatial feature enhancement (2D DWConv when H, W are provided, 1D otherwise)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=True)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
            groups=self.d_inner,
            bias=True)
        self.act = nn.SiLU()

        # SSM parameter projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Matrix A parameterization: A_log is initialized from contiguous 1..d_state
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Vector D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _ssm_scan(self, u):
        """Single-direction selective scan."""
        B, L, D = u.shape

        # x_proj outputs dt, B, C
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2 * d_state)
        delta = x_dbl[:, :, :self.dt_rank]
        B_mat = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_mat = x_dbl[:, :, self.dt_rank + self.d_state:]

        delta = F.softplus(self.dt_proj(delta))  # (B, L, D_inner)
        A = -torch.exp(self.A_log.float())       # (D_inner, N)

        if HAS_CUDA_SSM and u.is_cuda:
            # (B, D, L) for CUDA kernel
            u_t = u.transpose(1, 2).contiguous()
            delta_t = delta.transpose(1, 2).contiguous()
            B_t = B_mat.transpose(1, 2).contiguous()
            C_t = C_mat.transpose(1, 2).contiguous()
            y_t = selective_scan_fn(u_t, delta_t, A, B_t, C_t, self.D.float(), z=None,
                                    delta_bias=None, delta_softplus=False)
            return y_t.transpose(1, 2).contiguous()
        else:
            return selective_scan_pure_pytorch(u, delta, A, B_mat, C_mat, self.D)

    def forward(self, x, H=None, W=None):
        """Forward pass of Visual Mamba Block.

        Args:
            x (torch.Tensor): Input tokens of shape (B, N, C).
            H (int, optional): Spatial height.
            W (int, optional): Spatial width.

        Returns:
            torch.Tensor: Output tokens of shape (B, N, C).
        """
        residual = x
        x = self.norm(x)

        # In-projection to SSM branch (u) and gating branch (z)
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        # Local spatial convolution
        B, N, D_in = u.shape
        if H is not None and W is not None and H * W == N:
            u_conv = u.transpose(1, 2).view(B, D_in, H, W)
            u_conv = self.conv2d(u_conv)
            u_conv = u_conv.flatten(2).transpose(1, 2)
        else:
            u_conv = self.conv1d(u.transpose(1, 2)).transpose(1, 2)

        u = self.act(u_conv)

        # Visual Scanning (Bidirectional)
        y_fwd = self._ssm_scan(u)
        if self.bimamba:
            y_bwd = self._ssm_scan(u.flip(1)).flip(1)
            y = y_fwd + y_bwd
        else:
            y = y_fwd

        # Gating with SiLU(z)
        y = y * self.act(z)

        # Out-projection and residual connection
        out = self.out_proj(y)
        return residual + self.drop_path(out)

