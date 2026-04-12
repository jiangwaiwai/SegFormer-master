import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mmseg.datasets.color_utils import unique_rgb_colors
from mmseg.datasets.s2ds import S2DS_COLOR_TO_CLASS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect colors used in S2DS annotation masks.')
    parser.add_argument(
        '--data-root', default='data/S2DS', help='Root path of the S2DS dataset.')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to inspect.')
    parser.add_argument(
        '--files',
        nargs='*',
        default=[],
        help='Specific annotation filenames to inspect. Defaults to a sample set.')
    return parser.parse_args()


def main():
    args = parse_args()
    ann_dir = Path(args.data_root) / 'annotations' / args.split
    files = args.files or ['000_lab.png', '100_lab.png', '200_lab.png',
                           '300_lab.png', '400_lab.png', '500_lab.png']

    all_colors = set()
    for filename in files:
        path = ann_dir / filename
        if not path.exists():
            continue
        colors = unique_rgb_colors(np.array(Image.open(path)))
        print(f'\n{filename}:')
        for color in colors:
            marker = 'OK' if color in S2DS_COLOR_TO_CLASS else 'UNEXPECTED'
            print(f'  {color}  {marker}')
            all_colors.add(color)

    print('\nAll unique colors:')
    for color in sorted(all_colors):
        marker = 'OK' if color in S2DS_COLOR_TO_CLASS else 'UNEXPECTED'
        print(f'  {color}  {marker}')


if __name__ == '__main__':
    main()
