_base_ = ['./segformer.b0.1024x1024.s2ds.160k.segmamba.daeem.py']

# ==============================================================================
# 申报书消融实验组 ④：完整 D-SegMamba 模型
# 对照《大创申报书》第 20~23 页及第 25~26 页消融实验组 ④（图 17）设计：
# 1. 骨干网络：DSegMamba（双路径四阶段递进式混合编码器）
#    - 主干（Main）：四阶段混合 SegMamba
#      - Stage 1: 2 Mamba Blocks (32 通道, H/4 x W/4)
#      - Stage 2: 4 Mamba Blocks (64 通道, H/8 x W/8)
#      - Stage 3: 3 Transformer Blocks (160 通道, H/16 x W/16)
#      - Stage 4: 2 Mamba Blocks (256 通道, H/32 x W/32)
#    - 辅助分支（VR-Mamba）：4 个串联的 Mamba Block 逐级下采样
#      - Mamba Block 1: 32 通道, H/4 x W/4
#      - Mamba Block 2: 64 通道, H/8 x W/8
#      - Mamba Block 3: 160 通道, H/16 x W/16
#      - Mamba Block 4: 256 通道, H/32 x W/32
#    - 逐元素特征相加融合：F_i = Main_i + vr_gamma_i * Branch_i
#      - 零初始化 vr_gamma 参数确保冷启动承接组 ③ 43.53% 极高基石
# 2. 解码器：SegFormerDAEEMHead（方向感知边缘增强解码头）
#    - 浅层 1x7 水平、7x1 垂直、3x3 空洞卷积
#    - 显式 boundary_loss 辅助边界多任务监督 (loss_weight=0.2)
# ==============================================================================

model = dict(
    backbone=dict(
        type='DSegMamba',
        style='pytorch'))
