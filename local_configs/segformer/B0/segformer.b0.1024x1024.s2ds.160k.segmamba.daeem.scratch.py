_base_ = ['./segformer.b0.1024x1024.s2ds.160k.segmamba.daeem.py']

# ==============================================================================
# 申报书消融实验组 ③：SegMamba + DAEEM 从头训练配置 (Scratch)
# 严格受控实验原则：
# 1. pretrained=None（从零初始化，无任何 ImageNet 预训练权重）
# 2. 训练步数 (10,000 iters)、Batch Size (16)、学习率 (0.00024)、优化器与数据增强
#    与消融组 ① (SegFormer-B0 Scratch) 和消融组 ② (SegMamba Scratch) 100% 绝对对齐！
# ==============================================================================

model = dict(pretrained=None)
