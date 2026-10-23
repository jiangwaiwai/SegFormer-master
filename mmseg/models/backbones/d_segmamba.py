# ---------------------------------------------------------------
# D-SegMamba: Direction-Aware & Visual Residual Mamba-SegFormer
# Strictly implements the final full model described in the
# Project Application Document (Pages 20-23, Figures 9 & 13)
# Ablation Experiment Group 4: D-SegMamba (Full Model)
# ---------------------------------------------------------------
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
class DSegMamba(nn.Module):
    """D-SegMamba Dual-Path Backbone: SegMamba Main + VR-Mamba Branch.

    Strictly implements the hybrid dual-path architecture from the project proposal:
    1. Main Stream (SegMamba Encoder):
       - Stage 1: 2 Mamba Blocks (high-resolution detail & continuity)
       - Stage 2: 4 Mamba Blocks (mid-range context & broken crack association)
       - Stage 3: 3 Transformer Blocks (local semantic feature integration)
       - Stage 4: 2 Mamba Blocks (global dependency & long-range modeling)
    2. Parallel Visual Residual Mamba Branch (VR-Mamba):
       - Shares initial overlap patch embedding with Stage 1
       - 4 sequentially cascaded Mamba Blocks corresponding to Stages 1-4
       - Multi-scale downsampling synchronized with main stages
    3. Element-wise Feature Fusion (Mix Stage):
       - F_i = Main_i + vr_gamma_i * Branch_i  (i = 1, 2, 3, 4)
       - Zero-initialized learnable vr_gamma guarantees smooth initialization
         and non-negative transfer from Group 3 (43.53% mIoU baseline).
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
        super(DSegMamba, self).__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # Stochastic depth decay rule
        total_blocks = sum(depths) + 4  # 11 main blocks + 4 VR-Mamba branch blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # =====================================================================
        # 1. Main Backbone Stream (SegMamba)
        # =====================================================================

        # Main Stage 1: OverlapPatchEmbed 7x7 stride 4 -> 2 Mamba Blocks (dim=32)
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
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

        # Main Stage 2: OverlapPatchEmbed 3x3 stride 2 -> 4 Mamba Blocks (dim=64)
        cur += depths[0]
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
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

        # Main Stage 3: OverlapPatchEmbed 3x3 stride 2 -> 3 Transformer Blocks (dim=160)
        cur += depths[1]
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
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

        # Main Stage 4: OverlapPatchEmbed 3x3 stride 2 -> 2 Mamba Blocks (dim=256)
        cur += depths[2]
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
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

        # =====================================================================
        # 2. Parallel Visual Residual Mamba Branch (VR-Mamba)
        # =====================================================================
        cur += depths[3]

        # Branch Stage 1: Mamba Block 1 (processes shared H/4 x W/4 feature, dim=32)
        self.vr_block1 = MambaBlock(
            dim=embed_dims[0],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=dpr[cur + 0],
            norm_layer=norm_layer,
            bimamba=bimamba)
        self.vr_norm1 = norm_layer(embed_dims[0])

        # Branch Downsample 2: H/4 -> H/8, C: 32 -> 64
        self.vr_downsample2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        # Branch Stage 2: Mamba Block 2 (dim=64)
        self.vr_block2 = MambaBlock(
            dim=embed_dims[1],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=dpr[cur + 1],
            norm_layer=norm_layer,
            bimamba=bimamba)
        self.vr_norm2 = norm_layer(embed_dims[1])

        # Branch Downsample 3: H/8 -> H/16, C: 64 -> 160
        self.vr_downsample3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        # Branch Stage 3: Mamba Block 3 (dim=160)
        self.vr_block3 = MambaBlock(
            dim=embed_dims[2],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=dpr[cur + 2],
            norm_layer=norm_layer,
            bimamba=bimamba)
        self.vr_norm3 = norm_layer(embed_dims[2])

        # Branch Downsample 4: H/16 -> H/32, C: 160 -> 256
        self.vr_downsample4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        # Branch Stage 4: Mamba Block 4 (dim=256)
        self.vr_block4 = MambaBlock(
            dim=embed_dims[3],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=dpr[cur + 3],
            norm_layer=norm_layer,
            bimamba=bimamba)
        self.vr_norm4 = norm_layer(embed_dims[3])

        # =====================================================================
        # 3. Element-wise Feature Fusion (Active Element-wise Addition, init=1.0)
        # =====================================================================
        self.vr_gamma1 = nn.Parameter(torch.ones(1))
        self.vr_gamma2 = nn.Parameter(torch.ones(1))
        self.vr_gamma3 = nn.Parameter(torch.ones(1))
        self.vr_gamma4 = nn.Parameter(torch.ones(1))

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

        # -------------------------------------------------------------
        # Shared Initial Overlap Patch Embedding (H/4 x W/4, C=32)
        # -------------------------------------------------------------
        x_emb, H1, W1 = self.patch_embed1(x)

        # -------------------------------------------------------------
        # Stage 1: Main (2 Mamba) & VR Branch (Mamba Block 1)
        # -------------------------------------------------------------
        # Main Stage 1
        m1 = x_emb
        for blk in self.block1:
            m1 = blk(m1, H1, W1)
        m1 = self.norm1(m1)
        m1_spatial = m1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # VR-Mamba Branch Stage 1
        vr1 = self.vr_block1(x_emb, H1, W1)
        vr1 = self.vr_norm1(vr1)
        vr1_spatial = vr1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # Fusion 1 (Mix 1)
        f1 = m1_spatial + self.vr_gamma1 * vr1_spatial

        # -------------------------------------------------------------
        # Stage 2: Main (4 Mamba) & VR Branch (Mamba Block 2)
        # Cascaded closed-loop: Main Stage 2 receives Stage 1 fused features (f1)
        # -------------------------------------------------------------
        # Main Stage 2
        m2, H2, W2 = self.patch_embed2(f1)
        for blk in self.block2:
            m2 = blk(m2, H2, W2)
        m2 = self.norm2(m2)
        m2_spatial = m2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # VR-Mamba Branch Stage 2 (Cascaded from VR1)
        vr2, H2_vr, W2_vr = self.vr_downsample2(vr1_spatial)
        vr2 = self.vr_block2(vr2, H2_vr, W2_vr)
        vr2 = self.vr_norm2(vr2)
        vr2_spatial = vr2.reshape(B, H2_vr, W2_vr, -1).permute(0, 3, 1, 2).contiguous()

        # Fusion 2 (Mix 2)
        f2 = m2_spatial + self.vr_gamma2 * vr2_spatial

        # -------------------------------------------------------------
        # Stage 3: Main (3 Transformer) & VR Branch (Mamba Block 3)
        # Cascaded closed-loop: Main Stage 3 receives Stage 2 fused features (f2)
        # -------------------------------------------------------------
        # Main Stage 3
        m3, H3, W3 = self.patch_embed3(f2)
        for blk in self.block3:
            m3 = blk(m3, H3, W3)
        m3 = self.norm3(m3)
        m3_spatial = m3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # VR-Mamba Branch Stage 3 (Cascaded from VR2)
        vr3, H3_vr, W3_vr = self.vr_downsample3(vr2_spatial)
        vr3 = self.vr_block3(vr3, H3_vr, W3_vr)
        vr3 = self.vr_norm3(vr3)
        vr3_spatial = vr3.reshape(B, H3_vr, W3_vr, -1).permute(0, 3, 1, 2).contiguous()

        # Fusion 3 (Mix 3)
        f3 = m3_spatial + self.vr_gamma3 * vr3_spatial

        # -------------------------------------------------------------
        # Stage 4: Main (2 Mamba) & VR Branch (Mamba Block 4)
        # Cascaded closed-loop: Main Stage 4 receives Stage 3 fused features (f3)
        # -------------------------------------------------------------
        # Main Stage 4
        m4, H4, W4 = self.patch_embed4(f3)
        for blk in self.block4:
            m4 = blk(m4, H4, W4)
        m4 = self.norm4(m4)
        m4_spatial = m4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # VR-Mamba Branch Stage 4 (Cascaded from VR3)
        vr4, H4_vr, W4_vr = self.vr_downsample4(vr3_spatial)
        vr4 = self.vr_block4(vr4, H4_vr, W4_vr)
        vr4 = self.vr_norm4(vr4)
        vr4_spatial = vr4.reshape(B, H4_vr, W4_vr, -1).permute(0, 3, 1, 2).contiguous()

        # Fusion 4 (Mix 4)
        f4 = m4_spatial + self.vr_gamma4 * vr4_spatial

        return [f1, f2, f3, f4]

    def forward(self, x):
        return self.forward_features(x)
