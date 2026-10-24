_base_ = ['./segformer.b1.1024x1024.s2ds.160k.py']

# 从零初始化训练（Scratch），不加载任何 ImageNet 预训练权重
model = dict(pretrained=None)
