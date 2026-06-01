import json
import os.path as osp
import shutil

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook."""

    _default_greater_keys = {'mIoU', 'mAcc', 'aAcc', 'mDice', 'fg_mIoU', 'fg_mAcc', 'oIoU', 'P@0.5', 'P@0.6', 'P@0.7', 'P@0.8', 'P@0.9'}
    _default_less_keys = {'loss'}

    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.save_best = save_best
        self.rule = rule
        self.best_score = None
        self.key_indicator = save_best
        self.compare_func = None

        if self.save_best is not None:
            self.rule = self._infer_rule(self.rule, self.key_indicator)
            if self.rule == 'greater':
                self.compare_func = lambda x, y: x > y
            else:
                self.compare_func = lambda x, y: x < y

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_run(self, runner):
        if self.save_best is None or runner.meta is None:
            return

        hook_msgs = runner.meta.get('hook_msgs', {})
        best_eval_res = hook_msgs.get('best_eval_res')
        if not best_eval_res:
            return

        best_step_name = 'epoch' if self.by_epoch else 'iter'
        best_step = hook_msgs.get('best_step')
        best_ckpt = hook_msgs.get('best_ckpt')

        metric_parts = []
        for key, value in best_eval_res.items():
            metric_parts.append(f'{key}: {float(value):.4f}')
        metrics_text = ', '.join(metric_parts)

        runner.logger.info('Best validation summary:')
        runner.logger.info(
            f'best_{best_step_name}: {best_step}, {metrics_text}')
        runner.logger.info(f'best_checkpoint: {best_ckpt}')

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        self._save_best_checkpoint(runner, eval_res)

    def _save_best_checkpoint(self, runner, eval_res):
        if self.save_best is None:
            return

        if self.key_indicator not in eval_res:
            runner.logger.warning(
                f'save_best={self.key_indicator} is not in evaluation results: '
                f'{list(eval_res.keys())}')
            return

        if self.best_score is None:
            self.best_score = self._load_best_score(runner)

        score = float(eval_res[self.key_indicator])
        if not self.compare_func(score, self.best_score):
            return

        self.best_score = score
        best_ckpt_path = osp.join(runner.work_dir,
                                  f'best_{self.key_indicator}.pth')
        current_ckpt_path = self._get_ckpt_path(runner)
        shutil.copyfile(current_ckpt_path, best_ckpt_path)

        best_step = runner.epoch + 1 if self.by_epoch else runner.iter + 1
        best_eval_res = {
            key: float(val)
            for key, val in eval_res.items()
        }

        if runner.meta is not None:
            hook_msgs = runner.meta.setdefault('hook_msgs', {})
            hook_msgs['best_score'] = self.best_score
            hook_msgs['best_ckpt'] = best_ckpt_path
            hook_msgs['best_key'] = self.key_indicator
            hook_msgs['best_step'] = best_step
            hook_msgs['best_eval_res'] = best_eval_res

        self._dump_best_summary(runner, best_ckpt_path, best_step,
                                best_eval_res)

        runner.logger.info(
            f'Saved best checkpoint to {best_ckpt_path} '
            f'with {self.key_indicator}: {self.best_score:.4f}')

    def _dump_best_summary(self, runner, best_ckpt_path, best_step,
                           best_eval_res):
        summary = {
            'best_key': self.key_indicator,
            'best_score': float(self.best_score),
            'best_step_type': 'epoch' if self.by_epoch else 'iter',
            'best_step': int(best_step),
            'best_ckpt': best_ckpt_path,
            'metrics': best_eval_res
        }
        with open(osp.join(runner.work_dir, 'best_metrics.json'),
                  'w',
                  encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _load_best_score(self, runner):
        if runner.meta is not None:
            hook_msgs = runner.meta.get('hook_msgs', {})
            best_score = hook_msgs.get('best_score', None)
            if best_score is not None:
                return best_score
        return float('-inf') if self.rule == 'greater' else float('inf')

    def _get_ckpt_path(self, runner):
        if self.by_epoch:
            ckpt_name = f'epoch_{runner.epoch + 1}.pth'
        else:
            ckpt_name = f'iter_{runner.iter + 1}.pth'
        ckpt_path = osp.join(runner.work_dir, ckpt_name)
        if osp.isfile(ckpt_path):
            return ckpt_path
        return osp.join(runner.work_dir, 'latest.pth')

    def _infer_rule(self, rule, key_indicator):
        if rule is not None:
            if rule not in ['greater', 'less']:
                raise KeyError(f'rule must be greater or less, but got {rule}')
            return rule

        if key_indicator in self._default_greater_keys:
            return 'greater'
        if key_indicator in self._default_less_keys:
            return 'less'

        raise ValueError(
            'rule must be specified when save_best is not a known metric, '
            f'but got save_best={key_indicator}')


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.save_best = save_best
        self.rule = rule
        self.best_score = None
        self.key_indicator = save_best
        self.compare_func = None

        if self.save_best is not None:
            self.rule = self._infer_rule(self.rule, self.key_indicator)
            if self.rule == 'greater':
                self.compare_func = lambda x, y: x > y
            else:
                self.compare_func = lambda x, y: x < y

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

