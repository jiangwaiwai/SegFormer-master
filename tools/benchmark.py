import argparse
import os
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import apply_dataset_preset, list_dataset_presets


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--checkpoint', default=None, help='checkpoint file used for benchmarking')
    parser.add_argument(
        '--dataset',
        choices=list_dataset_presets(),
        help='dataset preset name to override dataset-related config')
    parser.add_argument(
        '--data-root',
        help='dataset root to use with --dataset, e.g. data/S2DS')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    dataset_name = args.dataset or os.environ.get('SEG_DATASET')
    data_root = args.data_root or os.environ.get('SEG_DATA_ROOT')
    if dataset_name is not None:
        apply_dataset_preset(cfg, dataset_name, data_root=data_root)
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        meta = checkpoint.get('meta', {})
        model.CLASSES = meta.get('CLASSES', dataset.CLASSES)
        model.PALETTE = meta.get('PALETTE', dataset.PALETTE)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    num_warmup = 5
    pure_inf_time = 0
    total_iters = 200

    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {total_iters}], '
                      f'fps: {fps:.2f} img / s')

        if (i + 1) == total_iters:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s')
            break


if __name__ == '__main__':
    main()
