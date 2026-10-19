import argparse
import os.path as osp
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmseg.models import build_segmentor
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args

def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


def get_tr_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    backbone = net.backbone
    extra_flops = 0
    stages = [('block1', 4), ('block2', 8), ('block3', 16), ('block4', 32)]
    for stage_name, scale in stages:
        blocks = getattr(backbone, stage_name, None)
        if blocks is not None and len(blocks) > 0:
            first_blk = blocks[0]
            if hasattr(first_blk, 'attn'):
                sr_ratio = getattr(first_blk.attn, 'sr_ratio', getattr(first_blk.attn, 'squeeze_ratio', 1))
                dim = getattr(first_blk.attn, 'dim', 64)
                num_heads = getattr(first_blk.attn, 'num_heads', 1)
                extra_flops += sra_flops(H // scale, W // scale, sr_ratio, dim, num_heads) * len(blocks)

    flops += extra_flops
    return flops_to_string(flops), params_to_string(params)


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).to(device)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    if hasattr(model.backbone, 'block1'):
        print('#### get transformer flops ####')
        with torch.no_grad():
            flops, params = get_tr_flops(model, input_shape)
    else:
        print('#### get CNN flops ####')
        flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
