_base_ = ['./segformer.b0.1024x1024.s2ds.160k.py']

model = dict(
    decode_head=dict(
        type='SegFormerDAEEMV2Head',
        daeem_v2_cfg=dict(
            kernel_size=7,
            dilation=2,
            reduction=4,
            max_groups=8,
        ),
        boundary_loss_weight=0.2,
        boundary_dilation=3,
    ))
