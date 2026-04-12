import copy


DATASET_PRESETS = {
    's2ds': {
        'dataset_type': 'S2DSDataset',
        'data_root': 'data/S2DS',
        'img_suffix': '.png',
        'seg_map_suffix': '_lab.png',
        'classes': (
            'background',
            'crack',
            'vegetation',
            'spalling',
            'corrosion',
            'efflorescence',
            'control_point',
        ),
        'palette': [
            [0, 0, 0],
            [255, 255, 255],
            [0, 255, 0],
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
        ],
    },
    'segmentation1641': {
        'dataset_type': 'Segmentation1641Dataset',
        'data_root': 'data/segmentation1641',
        'img_suffix': '.jpg',
        'seg_map_suffix': '_lab.png',
        'classes': (
            'background',
            'crack',
            'spall',
            'corrosion',
            'rebar',
        ),
        'palette': [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
        ],
    },
}


def list_dataset_presets():
    return sorted(DATASET_PRESETS.keys())


def _make_color_to_class(palette):
    return {tuple(color): idx for idx, color in enumerate(palette)}


def get_dataset_preset(name, data_root=None):
    key = name.lower()
    if key not in DATASET_PRESETS:
        raise KeyError(
            f'Unknown dataset preset: {name}. Available presets: {list_dataset_presets()}')
    preset = copy.deepcopy(DATASET_PRESETS[key])
    if data_root is not None:
        preset['data_root'] = data_root
    preset['name'] = key
    preset['color_to_class'] = _make_color_to_class(preset['palette'])
    return preset


def _set_num_classes(head_cfg, num_classes):
    if head_cfg is None:
        return
    if isinstance(head_cfg, list):
        for item in head_cfg:
            _set_num_classes(item, num_classes)
        return
    head_cfg['num_classes'] = num_classes


def _update_pipeline(pipeline, color_to_class):
    if pipeline is None:
        return
    for step in pipeline:
        if step.get('type') == 'LoadAnnotations':
            step['color_to_class'] = color_to_class
        if step.get('type') == 'MultiScaleFlipAug':
            _update_pipeline(step.get('transforms', []), color_to_class)


def _update_dataset_cfg(dataset_cfg, preset, split):
    target_cfg = dataset_cfg
    if dataset_cfg.get('type') == 'RepeatDataset':
        target_cfg = dataset_cfg['dataset']

    target_cfg['type'] = preset['dataset_type']
    target_cfg['data_root'] = preset['data_root']
    target_cfg['img_dir'] = f'images/{split}'
    target_cfg['ann_dir'] = f'annotations/{split}'
    target_cfg['img_suffix'] = preset['img_suffix']
    target_cfg['seg_map_suffix'] = preset['seg_map_suffix']
    target_cfg['classes'] = preset['classes']
    target_cfg['palette'] = preset['palette']
    target_cfg['color_to_class'] = preset['color_to_class']
    _update_pipeline(target_cfg.get('pipeline'), preset['color_to_class'])


def apply_dataset_preset(cfg, dataset_name, data_root=None):
    preset = get_dataset_preset(dataset_name, data_root=data_root)

    _set_num_classes(cfg.model.get('decode_head'), len(preset['classes']))
    _set_num_classes(cfg.model.get('auxiliary_head'), len(preset['classes']))

    _update_dataset_cfg(cfg.data.train, preset, 'train')
    _update_dataset_cfg(cfg.data.val, preset, 'val')
    _update_dataset_cfg(cfg.data.test, preset, 'test')

    cfg.dataset_name = preset['name']
    cfg.dataset_classes = preset['classes']
    cfg.dataset_palette = preset['palette']
    cfg.dataset_color_to_class = preset['color_to_class']
    return preset
