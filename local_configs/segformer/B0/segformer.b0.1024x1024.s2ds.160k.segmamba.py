_base_ = ['./segformer.b0.1024x1024.s2ds.160k.py']

# 申报书消融实验组 ②：纯 SegMamba（SegFormer-B0 + Mamba Block，不含 DAEEM，不含 VR-Mamba）
# 编码器 Stage 1: 2 Mamba Blocks
# 编码器 Stage 2: 4 Mamba Blocks
# 编码器 Stage 3: 3 Transformer Blocks (保留)
# 编码器 Stage 4: 2 Mamba Blocks
# 解码器: SegFormer 原生轻量级 MLP 解码头 (SegFormerHead)

model = dict(
    backbone=dict(
        type='SegMamba',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead'))

# 提高日志刷新频率（每 10 步打印一次，方便实时观察训练速度）
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ])


