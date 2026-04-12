import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = (
    'background',
    'crack',
    'spall',
    'corrosion',
    'rebar',
)

PALETTE = {
    'background': (0, 0, 0),
    'crack': (255, 255, 255),
    'spall': (255, 0, 0),
    'corrosion': (255, 255, 0),
    'rebar': (0, 0, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert segmentation1641 LabelMe annotations to a SegFormer-ready dataset layout.')
    parser.add_argument('data_root', help='Dataset root, e.g. D:/study/segmentation1641')
    parser.add_argument('--source-dir', default='before', help='Raw image/json directory name')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true', help='Overwrite existing split folders')
    return parser.parse_args()


def sanitize_stem(stem):
    if stem.startswith('(') and stem.endswith(')'):
        inner = stem[1:-1].strip()
        if inner.isdigit():
            return f'{int(inner):04d}'
    return stem.replace(' ', '_')


def load_annotation(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_size(annotation, image_path):
    height = annotation.get('imageHeight')
    width = annotation.get('imageWidth')
    if height and width:
        return width, height
    with Image.open(image_path) as img:
        return img.size


def rasterize_mask(annotation, image_size):
    mask = Image.new('RGB', image_size, PALETTE['background'])
    drawer = ImageDraw.Draw(mask)
    for shape in annotation.get('shapes', []):
        label = shape.get('label', '').strip()
        if label not in PALETTE:
            raise KeyError(f'Unknown label {label!r} in annotation.')
        points = shape.get('points', [])
        if len(points) < 2:
            continue
        xy = [tuple(point) for point in points]
        shape_type = shape.get('shape_type', 'polygon')
        if shape_type == 'rectangle' and len(xy) == 2:
            drawer.rectangle(xy, fill=PALETTE[label], outline=PALETTE[label])
        else:
            drawer.polygon(xy, fill=PALETTE[label], outline=PALETTE[label])
    return mask


def ensure_clean_dirs(root, force=False):
    for relative in (
        'images/train', 'images/val', 'images/test',
        'annotations/train', 'annotations/val', 'annotations/test',
    ):
        path = root / relative
        if path.exists() and any(path.iterdir()) and not force:
            raise FileExistsError(
                f'{path} already exists and is not empty. Use --force to overwrite.')
        path.mkdir(parents=True, exist_ok=True)


def clear_split_dirs(root):
    for relative in (
        'images/train', 'images/val', 'images/test',
        'annotations/train', 'annotations/val', 'annotations/test',
    ):
        path = root / relative
        if path.exists():
            shutil.rmtree(path)


def split_items(stems, train_ratio, val_ratio, seed):
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
        'test': shuffled[val_end:],
    }


def write_metadata(root, split_map, label_counter):
    metadata = {
        'classes': list(CLASSES),
        'palette': {key: list(value) for key, value in PALETTE.items()},
        'counts': {split: len(items) for split, items in split_map.items()},
        'label_shapes': dict(label_counter),
    }
    with open(root / 'dataset_meta.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    root = Path(args.data_root)
    source_dir = root / args.source_dir
    if not source_dir.is_dir():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    stems = []
    label_counter = Counter()
    for json_path in sorted(source_dir.glob('*.json')):
        image_path = source_dir / f'{json_path.stem}.jpg'
        if not image_path.exists():
            raise FileNotFoundError(f'Missing image for {json_path.name}')
        annotation = load_annotation(json_path)
        for shape in annotation.get('shapes', []):
            label_counter[shape.get('label', '').strip()] += 1
        stems.append(json_path.stem)

    unknown_labels = sorted(set(label_counter) - set(PALETTE))
    if unknown_labels:
        raise KeyError(f'Unsupported labels found: {unknown_labels}')

    if args.force:
        clear_split_dirs(root)
    ensure_clean_dirs(root, force=args.force)

    split_map = split_items(stems, args.train_ratio, args.val_ratio, args.seed)
    for split, split_stems in split_map.items():
        for stem in split_stems:
            json_path = source_dir / f'{stem}.json'
            image_path = source_dir / f'{stem}.jpg'
            annotation = load_annotation(json_path)
            image_size = get_image_size(annotation, image_path)
            clean_stem = sanitize_stem(stem)

            target_image = root / 'images' / split / f'{clean_stem}.jpg'
            target_mask = root / 'annotations' / split / f'{clean_stem}_lab.png'

            shutil.copy2(image_path, target_image)
            mask = rasterize_mask(annotation, image_size)
            mask.save(target_mask)

    write_metadata(root, split_map, label_counter)
    print('Prepared dataset at', root)
    for split, items in split_map.items():
        print(f'{split}: {len(items)}')


if __name__ == '__main__':
    main()
