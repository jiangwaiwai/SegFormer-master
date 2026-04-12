import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_SUFFIX = '_lab.png'
DEFAULT_SPLITS = ('train', 'val', 'test')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Resize a prepared semantic segmentation dataset offline.')
    parser.add_argument(
        'source_root',
        help='Prepared dataset root, e.g. /home/cqjtu/dataspace/sh/segmentation1641')
    parser.add_argument(
        'target_root',
        help='Output dataset root, e.g. /home/cqjtu/dataspace/sh/segmentation1641_1536')
    parser.add_argument(
        '--max-long-edge',
        type=int,
        default=1536,
        help='Resize only when the long edge is larger than this value.')
    parser.add_argument(
        '--img-quality',
        type=int,
        default=95,
        help='JPEG quality for output JPG images.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=list(DEFAULT_SPLITS),
        choices=DEFAULT_SPLITS,
        help='Dataset splits to process.')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only process the first N images of each split for a quick sanity check.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite target_root if it already exists.')
    return parser.parse_args()


def ensure_target_root(path, force=False):
    if path.exists():
        if not force:
            raise FileExistsError(
                f'{path} already exists. Use --force to overwrite it.')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_split_images(split_root):
    if not split_root.is_dir():
        raise FileNotFoundError(f'Missing image split directory: {split_root}')
    for path in sorted(split_root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def get_resample(name):
    if hasattr(Image, 'Resampling'):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def compute_target_size(size, max_long_edge):
    width, height = size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return width, height
    scale = max_long_edge / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height


def resize_image_file(src_path, dst_path, target_size, quality):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        resized = image.convert('RGB').resize(target_size, get_resample('BILINEAR'))
        save_kwargs = {}
        if dst_path.suffix.lower() in {'.jpg', '.jpeg'}:
            save_kwargs['quality'] = quality
        resized.save(dst_path, **save_kwargs)


def resize_mask_file(src_path, dst_path, target_size):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as mask:
        resized = mask.convert('RGB').resize(target_size, get_resample('NEAREST'))
        resized.save(dst_path)


def process_split(source_root, target_root, split, max_long_edge, img_quality, limit=None):
    src_image_dir = source_root / 'images' / split
    src_mask_dir = source_root / 'annotations' / split
    dst_image_dir = target_root / 'images' / split
    dst_mask_dir = target_root / 'annotations' / split
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    resized = 0
    size_samples = []

    for image_path in iter_split_images(src_image_dir):
        if limit is not None and processed >= limit:
            break

        stem = image_path.stem
        mask_path = src_mask_dir / f'{stem}{MASK_SUFFIX}'
        if not mask_path.exists():
            raise FileNotFoundError(f'Missing mask for {image_path.name}: {mask_path}')

        with Image.open(image_path) as image:
            original_size = image.size
        target_size = compute_target_size(original_size, max_long_edge)

        if target_size != original_size:
            resized += 1

        resize_image_file(
            image_path, dst_image_dir / image_path.name, target_size, img_quality)
        resize_mask_file(
            mask_path, dst_mask_dir / mask_path.name, target_size)

        processed += 1
        if len(size_samples) < 5:
            size_samples.append({
                'file': image_path.name,
                'original_size': list(original_size),
                'target_size': list(target_size),
            })

    return {
        'count': processed,
        'resized': resized,
        'samples': size_samples,
    }


def write_metadata(source_root, target_root, max_long_edge, split_stats, processed_splits, limit):
    source_meta_path = source_root / 'dataset_meta.json'
    metadata = {}
    if source_meta_path.exists():
        with open(source_meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    metadata['resized_from'] = str(source_root)
    metadata['resize'] = {
        'max_long_edge': max_long_edge,
        'splits': split_stats,
        'processed_splits': list(processed_splits),
        'limit_per_split': limit,
    }

    with open(target_root / 'dataset_meta.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f'Source dataset root not found: {source_root}')

    ensure_target_root(target_root, force=args.force)

    split_stats = {}
    for split in args.splits:
        split_stats[split] = process_split(
            source_root,
            target_root,
            split,
            max_long_edge=args.max_long_edge,
            img_quality=args.img_quality,
            limit=args.limit)

    write_metadata(
        source_root,
        target_root,
        args.max_long_edge,
        split_stats,
        args.splits,
        args.limit)

    print(f'Resized dataset written to: {target_root}')
    for split in args.splits:
        stats = split_stats[split]
        print(
            f'{split}: count={stats["count"]}, resized={stats["resized"]}, '
            f'kept={stats["count"] - stats["resized"]}')


if __name__ == '__main__':
    main()
