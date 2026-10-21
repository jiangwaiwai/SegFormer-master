_base_ = ['./segformer.b0.1024x1024.s2ds.160k.segmamba.py']

# ==============================================================================
# 申报书消融实验组 ③：SegMamba + DAEEM 模块
# 对照《大创申报书》第 25~26 页 消融实验组 ③（图 17）设计：
# 1. 骨干网络：SegMamba（四阶段 Mamba-Transformer 混合编码器，继承自 segmamba.py）
#    - Stage 1: 2 Mamba Blocks (32 通道, H/4 x W/4)
#    - Stage 2: 4 Mamba Blocks (64 通道, H/8 x W/8)
#    - Stage 3: 3 Transformer Blocks (160 通道, H/16 x W/16)
#    - Stage 4: 2 Mamba Blocks (256 通道, H/32 x W/32)
# 2. 解码器：SegFormerDAEEMHead（方向感知边缘增强解码头，实现方案一）
#    - 浅层 1x7 水平、7x1 垂直、3x3 空洞方向感知卷积核
#    - 自适应分支选择器 + 遮挡鲁棒性置信度门控
#    - 零初始化残差系数 shallow_gamma，保障正向增益
#    - 显式 boundary_loss 辅助边界多任务监督 (loss_weight=0.2)
# ==============================================================================

model = dict(
    decode_head=dict(
        type='SegFormerDAEEMHead',
        daeem_cfg=dict(
            kernel_size=7,
            dilation=2,
            reduction=4,
            max_groups=8,
        ),
        boundary_loss_weight=0.2,
        boundary_dilation=3,
    ))
