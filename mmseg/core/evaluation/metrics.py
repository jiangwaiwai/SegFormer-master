import mmcv
import numpy as np


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union."""

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', channel_order='rgb', backend='pillow')
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union."""

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float64)
    total_area_union = np.zeros((num_classes, ), dtype=np.float64)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float64)
    total_area_label = np.zeros((num_classes, ), dtype=np.float64)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def foreground_intersect_and_union(pred_label,
                                   label,
                                   ignore_index,
                                   background_index=0,
                                   label_map=dict(),
                                   reduce_zero_label=False):
    """Calculate foreground-vs-background intersection and union."""

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', channel_order='rgb', backend='pillow')
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    pred_fg = pred_label != background_index
    label_fg = label != background_index

    intersection = np.logical_and(pred_fg, label_fg).sum(dtype=np.float64)
    union = np.logical_or(pred_fg, label_fg).sum(dtype=np.float64)
    return intersection, union


def eval_object_metrics(results,
                        gt_seg_maps,
                        ignore_index,
                        background_index=0,
                        iou_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
                        label_map=dict(),
                        reduce_zero_label=False):
    """Evaluate foreground object metrics compatible with MAFN.

    All non-background classes are merged into a single foreground mask.
    P@t measures the ratio of samples whose foreground IoU is >= t.
    oIoU is the global IoU computed from accumulated intersections/unions.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    seg_correct = np.zeros((len(iou_thresholds), ), dtype=np.float64)
    seg_total = 0
    cum_intersection = 0.0
    cum_union = 0.0

    for i in range(num_imgs):
        intersection, union = foreground_intersect_and_union(
            results[i],
            gt_seg_maps[i],
            ignore_index,
            background_index=background_index,
            label_map=label_map,
            reduce_zero_label=reduce_zero_label)

        if intersection == 0 or union == 0:
            iou = 0.0
        else:
            iou = float(intersection) / float(union)

        cum_intersection += intersection
        cum_union += union
        for idx, threshold in enumerate(iou_thresholds):
            seg_correct[idx] += float(iou >= threshold)
        seg_total += 1

    if seg_total == 0:
        return {f'P@{thr:.1f}': 0.0 for thr in iou_thresholds} | {'oIoU': 0.0}

    object_metrics = {}
    for idx, threshold in enumerate(iou_thresholds):
        object_metrics[f'P@{threshold:.1f}'] = seg_correct[idx] / seg_total

    if cum_intersection == 0 or cum_union == 0:
        object_metrics['oIoU'] = 0.0
    else:
        object_metrics['oIoU'] = float(cum_intersection) / float(cum_union)
    return object_metrics


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)."""

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)."""

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics."""

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics
