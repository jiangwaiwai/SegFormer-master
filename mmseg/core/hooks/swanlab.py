import numbers
import os.path as osp

import mmcv
import numpy as np
from mmcv.runner import HOOKS, Hook

from mmseg.datasets.color_utils import ensure_label_map


@HOOKS.register_module()
class SwanlabHook(Hook):
    """Log training metrics and validation visualizations to SwanLab."""

    priority = 'VERY_LOW'

    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 by_epoch=False,
                 image_interval=None,
                 num_eval_images=4,
                 image_size=768):
        self.init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.interval = interval
        self.by_epoch = by_epoch
        self.image_interval = image_interval
        self.num_eval_images = num_eval_images
        self.image_size = image_size
        self.eval_dataset = None
        self.run = None
        self.swanlab = None

    def before_run(self, runner):
        if getattr(runner, 'rank', 0) != 0:
            return

        try:
            import swanlab
        except ImportError as exc:
            raise ImportError(
                'SwanLab is enabled but not installed. Run `pip install swanlab`.') from exc

        self.swanlab = swanlab
        self.eval_dataset = self._find_eval_dataset(runner)

        init_kwargs = self.init_kwargs.copy()
        init_kwargs.setdefault('project', 'mmsegmentation')
        init_kwargs.setdefault('experiment_name',
                               osp.basename(osp.abspath(runner.work_dir)))
        init_kwargs.setdefault('config', self._build_run_config(runner))
        self.run = swanlab.init(**init_kwargs)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return
        self._log_at_step(runner, runner.iter + 1)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self._log_at_step(runner, runner.epoch + 1)

    def after_run(self, runner):
        if self.run is not None:
            self.run.finish()

    def _log_at_step(self, runner, step):
        if self.run is None:
            return

        should_log_metrics = False
        if self.interval:
            if self.by_epoch:
                should_log_metrics = self.every_n_epochs(runner, self.interval)
            else:
                should_log_metrics = self.every_n_iters(runner, self.interval)
        if should_log_metrics:
            self._log_metrics(runner, step)

        should_log_images = False
        if self.image_interval:
            if self.by_epoch:
                should_log_images = self.every_n_epochs(runner, self.image_interval)
            else:
                should_log_images = self.every_n_iters(runner, self.image_interval)
        if should_log_images:
            self._log_eval_visualizations(runner, step)

    def _log_metrics(self, runner, step):
        log_output = getattr(runner.log_buffer, 'output', None)
        if not log_output:
            return

        payload = {}
        for key, value in log_output.items():
            scalar = self._to_scalar(value)
            if scalar is None:
                continue
            payload[self._format_metric_name(key)] = scalar

        if payload:
            self.run.log(payload, step=step)

    def _log_eval_visualizations(self, runner, step):
        dataset = self.eval_dataset or self._find_eval_dataset(runner)
        if dataset is None or len(dataset) == 0:
            return
        if getattr(dataset, 'ann_dir', None) is None or getattr(dataset, 'PALETTE', None) is None:
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if not hasattr(model, 'cfg'):
            return

        image_count = min(self.num_eval_images, len(dataset))
        if image_count <= 0:
            return

        sample_indices = np.linspace(0, len(dataset) - 1, num=image_count, dtype=int)
        visuals = []
        for idx in sample_indices.tolist():
            visual = self._build_sample_visual(model, dataset, idx)
            if visual is not None:
                visuals.append(
                    self.swanlab.Image(
                        visual['image'],
                        caption=visual['caption']))

        if visuals:
            self.run.log({'val/visualizations': visuals}, step=step)

    def _build_sample_visual(self, model, dataset, idx):
        from mmseg.apis.inference import inference_segmentor

        img_info = dataset.img_infos[idx]
        img_path = osp.join(dataset.img_dir, img_info['filename'])
        ann_path = osp.join(dataset.ann_dir, img_info['ann']['seg_map'])

        image = mmcv.imread(img_path)
        if image is None:
            return None

        was_training = model.training
        model.eval()
        try:
            pred = inference_segmentor(model, img_path)[0]
        finally:
            if was_training:
                model.train()

        gt_mask = ensure_label_map(
            ann_path,
            getattr(dataset, 'COLOR_TO_CLASS', None),
            getattr(dataset, 'ignore_index', 255))

        image_rgb = mmcv.bgr2rgb(image)
        gt_overlay = self._overlay_seg_map(
            image_rgb, gt_mask, dataset.PALETTE, getattr(dataset, 'ignore_index', 255))
        pred_overlay = self._overlay_seg_map(image_rgb, pred, dataset.PALETTE)

        separator = np.full((image_rgb.shape[0], 12, 3), 255, dtype=np.uint8)
        panel = np.concatenate(
            [image_rgb, separator, gt_overlay, separator, pred_overlay], axis=1)
        if self.image_size is not None:
            panel = mmcv.imrescale(panel, (self.image_size, self.image_size))

        return dict(
            image=panel,
            caption=f"{img_info['filename']} | left=input middle=gt right=pred")

    def _overlay_seg_map(self, image_rgb, seg_map, palette, ignore_index=None):
        color_seg = np.zeros_like(image_rgb)
        for label, color in enumerate(palette):
            color_seg[seg_map == label] = color

        if ignore_index is not None:
            valid_mask = seg_map != ignore_index
        else:
            valid_mask = np.ones(seg_map.shape, dtype=bool)

        overlay = image_rgb.copy()
        overlay[valid_mask] = (
            image_rgb[valid_mask].astype(np.float32) * 0.5
            + color_seg[valid_mask].astype(np.float32) * 0.5).astype(np.uint8)
        return overlay

    def _find_eval_dataset(self, runner):
        for hook in runner.hooks:
            dataloader = getattr(hook, 'dataloader', None)
            if dataloader is not None:
                return dataloader.dataset
        return None

    def _build_run_config(self, runner):
        meta = runner.meta or {}
        config = dict(work_dir=runner.work_dir)
        if meta.get('exp_name') is not None:
            config['config_file'] = meta['exp_name']
        if meta.get('seed') is not None:
            config['seed'] = meta['seed']

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        classes = getattr(model, 'CLASSES', None)
        if classes is not None:
            config['classes'] = list(classes)
            config['num_classes'] = len(classes)
        return config

    def _format_metric_name(self, key):
        if key in {'mIoU', 'mDice', 'mAcc', 'aAcc', 'fg_mIoU', 'fg_mAcc', 'oIoU', 'P@0.5', 'P@0.6', 'P@0.7', 'P@0.8', 'P@0.9'}:
            return f'val/{key}'
        if key == 'lr':
            return 'train/lr'
        if key in {'time', 'data_time', 'memory'}:
            return f'system/{key}'
        return f'train/{key}'

    def _to_scalar(self, value):
        if isinstance(value, np.generic):
            return float(value)
        if isinstance(value, numbers.Number):
            return float(value)
        if hasattr(value, 'item'):
            try:
                return float(value.item())
            except (TypeError, ValueError):
                return None
        return None

