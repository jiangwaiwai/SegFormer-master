_base_ = ['./segformer.b0.1024x1024.s2ds.160k.py']

model = dict(
    backbone=dict(
        use_daeem=True,
        daeem_stages=(1,),
        daeem_cfg=dict(
            kernel_size=7,
            dilation=2,
            reduction=4,
        )))
