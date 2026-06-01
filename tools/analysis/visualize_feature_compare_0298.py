import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from PIL import Image

matplotlib.use("Agg")

BRIDGESEG_CLASSES = (
    "background",
    "crack",
    "spall",
    "corrosion",
    "rebar",
)

BRIDGESEG_PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [255, 255, 0],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

IMAGE_PATH = "/home/cqjtu/dataspace/sh/segmentation1641_1536/images/test/0298.jpg"

BASELINE_CKPT = (
    "/home/cqjtu/dataspace/sh/SegFormer/work_dirs/"
    "segformer_b0_seg1641_scratch √/best_mIoU.pth"
)
DAEEM_CKPT = (
    "/home/cqjtu/dataspace/sh/SegFormer/work_dirs/"
    "segformer_b0_seg1641_daeem_scratch_gpu0 √/best_mIoU.pth"
)

OUTPUT_PATH = (
    "/home/cqjtu/dataspace/sh/SegFormer/work_dirs/"
    "feature_compare_0298_baseline_vs_daeem.png"
)

BASELINE_CONFIG_TEXT = """
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mit_b0',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
""".strip()

DAEEM_CONFIG_TEXT = """
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mit_b0',
        style='pytorch',
        use_daeem=True,
        daeem_stages=(1,),
        daeem_cfg=dict(
            kernel_size=7,
            dilation=2,
            reduction=4,
        )),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
""".strip()


def write_temp_config(config_text, filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as file:
        file.write(config_text + "\n")
    return path


def build_model(config_text, checkpoint_path, device, tmp_name):
    config_path = write_temp_config(config_text, tmp_name)
    cfg = Config.fromfile(config_path)

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False)

    model.CLASSES = BRIDGESEG_CLASSES
    model.PALETTE = BRIDGESEG_PALETTE.tolist()
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def prepare_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_float = image_np.astype(np.float32)
    image_norm = (image_float - MEAN) / STD
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    img_meta = [[{
        "filename": image_path,
        "ori_filename": os.path.basename(image_path),
        "ori_shape": image_np.shape,
        "img_shape": image_np.shape,
        "pad_shape": image_np.shape,
        "scale_factor": 1.0,
        "flip": False,
        "flip_direction": None,
        "img_norm_cfg": {
            "mean": MEAN,
            "std": STD,
            "to_rgb": True,
        },
    }]]
    return image_np, image_tensor, img_meta


def feature_to_heatmap(feature, out_hw):
    feature_map = feature.detach().abs().mean(dim=1, keepdim=True)
    feature_map = F.interpolate(
        feature_map,
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    )
    feature_map = feature_map[0, 0].cpu().numpy()
    feature_map = feature_map - feature_map.min()
    if feature_map.max() > 1e-8:
        feature_map = feature_map / feature_map.max()
    return feature_map


def overlay_prediction(image_np, pred):
    color_mask = BRIDGESEG_PALETTE[pred]
    overlay = image_np.astype(np.float32).copy()
    mask = pred != 0
    overlay[mask] = overlay[mask] * 0.55 + color_mask[mask] * 0.45
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_model(model, image_tensor, img_meta, out_hw):
    with torch.no_grad():
        features = model.extract_feat(image_tensor)
        seg_logit = model.encode_decode(image_tensor, img_meta[0])
        pred = seg_logit.argmax(dim=1)[0].cpu().numpy()

    heatmaps = [
        feature_to_heatmap(features[3], out_hw),
        feature_to_heatmap(features[2], out_hw),
        feature_to_heatmap(features[1], out_hw),
        feature_to_heatmap(features[0], out_hw),
    ]
    return heatmaps, pred


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_np, image_tensor, img_meta = prepare_image(IMAGE_PATH, device)
    out_hw = image_np.shape[:2]

    baseline_model = build_model(
        BASELINE_CONFIG_TEXT,
        BASELINE_CKPT,
        device,
        "seg1641_baseline_vis.py",
    )
    daeem_model = build_model(
        DAEEM_CONFIG_TEXT,
        DAEEM_CKPT,
        device,
        "seg1641_daeem_vis.py",
    )

    baseline_heatmaps, baseline_pred = run_model(
        baseline_model,
        image_tensor,
        img_meta,
        out_hw,
    )
    daeem_heatmaps, daeem_pred = run_model(
        daeem_model,
        image_tensor,
        img_meta,
        out_hw,
    )

    baseline_overlay = overlay_prediction(image_np, baseline_pred)
    daeem_overlay = overlay_prediction(image_np, daeem_pred)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6), dpi=200)
    titles = ["Input", r"$F_e^4$", r"$F_e^3$", r"$F_e^2$", r"$F_e^1$", "Prediction"]
    row_names = ["baseline", "+DAEEM"]

    for index, title in enumerate(titles):
        axes[0, index].set_title(title, fontsize=16, fontweight="bold")

    axes[0, 0].imshow(image_np)
    axes[0, 1].imshow(baseline_heatmaps[0], cmap="viridis")
    axes[0, 2].imshow(baseline_heatmaps[1], cmap="viridis")
    axes[0, 3].imshow(baseline_heatmaps[2], cmap="viridis")
    axes[0, 4].imshow(baseline_heatmaps[3], cmap="viridis")
    axes[0, 5].imshow(baseline_overlay)

    axes[1, 0].imshow(image_np)
    axes[1, 1].imshow(daeem_heatmaps[0], cmap="viridis")
    axes[1, 2].imshow(daeem_heatmaps[1], cmap="viridis")
    axes[1, 3].imshow(daeem_heatmaps[2], cmap="viridis")
    axes[1, 4].imshow(daeem_heatmaps[3], cmap="viridis")
    axes[1, 5].imshow(daeem_overlay)

    for row in range(2):
        axes[row, 0].set_ylabel(row_names[row], fontsize=18, rotation=90, labelpad=18)
        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    plt.subplots_adjust(
        wspace=0.02,
        hspace=0.04,
        left=0.06,
        right=0.995,
        top=0.90,
        bottom=0.02,
    )
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
