import argparse
import csv
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import apply_dataset_preset, list_dataset_presets


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--dataset',
        choices=list_dataset_presets(),
        help='dataset preset name to override dataset-related config')
    parser.add_argument(
        '--data-root',
        help='dataset root to use with --dataset, e.g. data/S2DS')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted overlay images will be saved')
    parser.add_argument(
        '--mask-dir', help='directory where pure color prediction masks will be saved')
    parser.add_argument(
        '--csv-out', help='CSV file used to append one row per test run')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def append_metrics_csv(csv_path, row):
    csv_dir = osp.dirname(csv_path)
    if csv_dir:
        mmcv.mkdir_or_exist(csv_dir)
    file_exists = osp.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.mask_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir" or "--mask-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    dataset_name = args.dataset or os.environ.get('SEG_DATASET')
    data_root = args.data_root or os.environ.get('SEG_DATA_ROOT')
    if dataset_name is not None:
        apply_dataset_preset(cfg, dataset_name, data_root=data_root)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if distributed and args.mask_dir:
        raise ValueError('--mask-dir is only supported in single-GPU test mode')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    meta = checkpoint.get('meta', {})
    model.CLASSES = meta.get('CLASSES', dataset.CLASSES)
    model.PALETTE = meta.get('PALETTE', dataset.PALETTE)

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            show=False,
            out_dir=args.show_dir,
            mask_dir=args.mask_dir,
            efficient_test=efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_results = dataset.evaluate(outputs, args.eval, **kwargs)
            csv_out = args.csv_out
            if csv_out is None:
                csv_out = osp.join(osp.dirname(args.checkpoint), 'test_metrics.csv')
            row = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'config': osp.abspath(args.config),
                'checkpoint': osp.abspath(args.checkpoint),
                'dataset': dataset_name or cfg.data.test.type,
                'data_root': data_root or getattr(dataset, 'data_root', ''),
                'show_dir': args.show_dir or '',
                'mask_dir': args.mask_dir or '',
            }
            for key in sorted(eval_results.keys()):
                row[key] = eval_results[key]
            append_metrics_csv(csv_out, row)
            print(f'\nmetrics appended to {csv_out}')


if __name__ == '__main__':
    main()
