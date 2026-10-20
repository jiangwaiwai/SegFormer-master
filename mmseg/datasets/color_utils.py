import numpy as np


def _strip_alpha_channel(mask):
    """Drop the alpha channel from an RGBA segmentation mask."""
    if mask.ndim == 3 and mask.shape[2] == 4:
        return mask[:, :, :3]
    return mask


def color_mask_to_label_map(mask, color_to_class, ignore_index=255):
    """Convert an RGB(A) segmentation mask to a label map."""
    mask = np.asarray(mask)
    mask = _strip_alpha_channel(mask)

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(
            f'Expected an RGB(A) segmentation mask, got shape {mask.shape}.')

    # Fail-safe BGR auto-correction:
    # If the dataset expects Yellow (255, 255, 0) (e.g. segmentation1641 corrosion),
    # but the array was read via OpenCV/MMCV default (BGR) where Yellow becomes Cyan [0, 255, 255],
    # detect this and invert channels back to RGB.
    if (255, 255, 0) in color_to_class:
        has_bgr_yellow = np.any((mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))
        has_rgb_yellow = np.any((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0))
        if has_bgr_yellow and not has_rgb_yellow:
            mask = mask[:, :, ::-1]

    label_map = np.full(mask.shape[:2], ignore_index, dtype=np.uint8)
    for color, class_id in color_to_class.items():
        label_map[np.all(mask == color, axis=-1)] = class_id
    return label_map


def ensure_label_map(mask, color_to_class=None, ignore_index=255):
    """Return a 2D uint8 label map from either index or RGB(A) annotations."""
    if isinstance(mask, str):
        from PIL import Image
        with Image.open(mask) as _img:
            if color_to_class is not None:
                mask = np.array(_img.convert('RGB'))
            else:
                mask = np.array(_img)
    elif hasattr(mask, 'convert'):
        if color_to_class is not None:
            mask = np.array(mask.convert('RGB'))
        else:
            mask = np.array(mask)
    else:
        mask = np.asarray(mask)

    if mask.ndim == 2:
        return mask.astype(np.uint8)

    mask = _strip_alpha_channel(mask)

    if mask.ndim == 3 and mask.shape[2] == 1:
        return mask[:, :, 0].astype(np.uint8)

    if color_to_class is None:
        raise ValueError(
            'A color segmentation map was loaded, but no color_to_class '
            'mapping was provided.')

    return color_mask_to_label_map(mask, color_to_class, ignore_index)


def unique_rgb_colors(mask):
    """Return the sorted unique RGB colors present in a segmentation mask."""
    mask = np.asarray(mask)
    mask = _strip_alpha_channel(mask)

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(
            f'Expected an RGB(A) segmentation mask, got shape {mask.shape}.')

    colors = np.unique(mask.reshape(-1, 3), axis=0)
    return [tuple(int(channel) for channel in color) for color in colors]
