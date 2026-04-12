#!/usr/bin/env python
"""Convert S2DS RGB annotations to label maps."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mmseg.datasets.color_utils import color_mask_to_label_map
from mmseg.datasets.s2ds import S2DS_COLOR_TO_CLASS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert S2DS RGB annotations to label maps.')
    parser.add_argument(
        '--data-root', default='data/S2DS', help='Root path of the S2DS dataset.')
    parser.add_argument(
        '--input-dir',
        default='annotations',
        help='Relative path to the source annotation directory.')
    parser.add_argument(
        '--output-dir',
        default='annotations_converted',
        help='Relative path to the output annotation directory.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to convert.')
    return parser.parse_args()


def convert_split(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob('*_lab.png'))

    print(f'\nConverting {input_dir} -> {output_dir}')
    for path in tqdm(files):
        class_img = color_mask_to_label_map(
            np.array(Image.open(path)), S2DS_COLOR_TO_CLASS)
        Image.fromarray(class_img).save(output_dir / path.name)

    print(f'Converted {len(files)} files.')


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print('=' * 60)
    print('S2DS annotation conversion')
    print('=' * 60)
    for split in args.splits:
        convert_split(
            data_root / args.input_dir / split,
            data_root / args.output_dir / split)

    print('\nDone.')


if __name__ == '__main__':
    main()
