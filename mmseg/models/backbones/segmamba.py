import math
import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import trunc_normal_
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint

from .mix_transformer import Block, OverlapPatchEmbed
from mmseg.models.utils.mamba_block import MambaBlock


@BACKBONES.register_module()
class SegMamba(nn.Module):
    """SegMamba Backbone based on SegFormer-B0.

    Strictly implements the hybrid encoder architecture described in the project
    application document:
        - Stage 1: 2 Mamba Blocks (high-resolution detail & continuity)
        - Stage 2: 4 Mamba Blocks (mid-range context & broken crack association)
        - Stage 3: 3 Transformer Blocks (local semantic feature integration via self-attention)
        - Stage 4: 2 Mamba Blocks (global dependency & long-range modeling)

    Args:
        img_size (int): Base image input size. Default: 224.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dims (list[int]): Channel dimensions for the 4 stages. Default: [32, 64, 160, 256].
        num_heads (list[int]): Attention heads for each stage. Default: [1, 2, 5, 8].
        mlp_ratios (list[int]): MLP expansion ratios for each stage. Default: [4, 4, 4, 4].
        qkv_bias (bool): Whether to include bias in QKV projection. Default: True.
        qk_scale (float, optional): Custom scaling factor for attention. Default: None.
        drop_rate (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer (nn.Module): Normalization layer. Default: LayerNorm with eps=1e-6.
        depths (list[int]): Depths for each of the 4 stages. Default: [2, 4, 3, 2].
        sr_ratios (list[int]): Spatial reduction ratios for attention. Default: [8, 4, 2, 1].
        d_state (int): SSM state dimension for Mamba blocks. Default: 16.
        d_conv (int): 1D/2D convolution kernel size for Mamba blocks. Default: 4.
        expand (int): Channel expansion ratio for Mamba blocks. Default: 2.
        bimamba (bool): Whether to use bidirectional scanning. Default: True.
        style (str): Weight initialization style. Default: 'pytorch'.
    """

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 4, 3, 2],
                 sr_ratios=[8, 4, 2, 1],
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 bimamba=True,
                 style='pytorch',
                 **kwargs):
        super(SegMamba, self).__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # Stochastic depth decay rule across all 11 blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ------------------ Stage 1 ------------------
        # Patch Embed: H/4 x W/4, C=32
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        # 2 Mamba Blocks
        cur = 0
        self.block1 = nn.ModuleList([
            MambaBlock(
                dim=embed_dims[0],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                bimamba=bimamba) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # ------------------ Stage 2 ------------------
        # Patch Embed: H/8 x W/8, C=64
        cur += depths[0]
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        # 4 Mamba Blocks
        self.block2 = nn.ModuleList([
            MambaBlock(
                dim=embed_dims[1],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                bimamba=bimamba) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # ------------------ Stage 3 ------------------
        # Patch Embed: H/16 x W/16, C=160
        cur += depths[1]
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        # 3 Transformer Blocks (Retained per application specifications)
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # ------------------ Stage 4 ------------------
        # Patch Embed: H/32 x W/32, C=256
        cur += depths[2]
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        # 2 Mamba Blocks
        self.block4 = nn.ModuleList([
            MambaBlock(
                dim=embed_dims[3],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                bimamba=bimamba) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1: 2 Mamba blocks
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2: 4 Mamba blocks
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3: 3 Transformer blocks
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4: 2 Mamba blocks
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        return self.forward_features(x)

