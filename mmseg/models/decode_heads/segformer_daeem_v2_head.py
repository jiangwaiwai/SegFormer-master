import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import DAEEMV2
from .decode_head import BaseDecodeHead
from .segformer_head import MLP


@HEADS.register_module()
class SegFormerDAEEMV2Head(BaseDecodeHead):
    """SegFormer head with a shallow boundary enhancement side branch."""

    def __init__(self,
                 feature_strides,
                 daeem_v2_cfg=None,
                 boundary_loss_weight=0.2,
                 boundary_dilation=3,
                 **kwargs):
        super(SegFormerDAEEMV2Head, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_dilation = boundary_dilation

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_edge = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.daeem_v2 = DAEEMV2(
            in_channels=c1_in_channels,
            **({} if daeem_v2_cfg is None else dict(daeem_v2_cfg)))
        self.shallow_fuse = ConvModule(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.shallow_gamma = nn.Parameter(torch.zeros(1))

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def _forward_all(self, inputs):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n = c4.shape[0]

        edge_feat, boundary_logits = self.daeem_v2(c1)

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear',
                     align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear',
                     align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear',
                     align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])
        _edge = self.linear_edge(edge_feat).permute(0, 2, 1).reshape(
            n, -1, edge_feat.shape[2], edge_feat.shape[3])
        edge_gate = torch.sigmoid(boundary_logits)
        _edge = _edge * edge_gate
        _edge = self.shallow_fuse(torch.cat([_c1, _edge], dim=1))
        _c1 = _c1 + self.shallow_gamma * _edge

        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        seg_logits = self.linear_pred(self.dropout(fused))
        return seg_logits, boundary_logits

    def forward(self, inputs):
        seg_logits, _ = self._forward_all(inputs)
        return seg_logits

    def _build_boundary_targets(self, seg_label):
        target = seg_label.squeeze(1)
        ignore_mask = target.eq(self.ignore_index)
        boundary = torch.zeros_like(target, dtype=torch.float32)

        horizontal = (target[:, :, 1:] != target[:, :, :-1])
        horizontal = horizontal & (~ignore_mask[:, :, 1:]) & (~ignore_mask[:, :, :-1])
        boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], horizontal.float())
        boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], horizontal.float())

        vertical = (target[:, 1:, :] != target[:, :-1, :])
        vertical = vertical & (~ignore_mask[:, 1:, :]) & (~ignore_mask[:, :-1, :])
        boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], vertical.float())
        boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], vertical.float())

        boundary = boundary.unsqueeze(1)
        if self.boundary_dilation and self.boundary_dilation > 1:
            boundary = F.max_pool2d(
                boundary,
                kernel_size=self.boundary_dilation,
                stride=1,
                padding=self.boundary_dilation // 2)
        valid_mask = (~ignore_mask).unsqueeze(1).float()
        return boundary, valid_mask

    def boundary_losses(self, boundary_logit, seg_label):
        boundary_logit = resize(
            input=boundary_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        boundary_target, valid_mask = self._build_boundary_targets(seg_label)
        loss = F.binary_cross_entropy_with_logits(
            boundary_logit, boundary_target, reduction='none')
        loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        return {'loss_boundary': loss * self.boundary_loss_weight}

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, boundary_logits = self._forward_all(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        losses.update(self.boundary_losses(boundary_logits, gt_semantic_seg))
        return losses
