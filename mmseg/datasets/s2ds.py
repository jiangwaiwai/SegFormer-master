from .builder import DATASETS
from .custom import CustomDataset


S2DS_CLASSES = (
    'background',
    'crack',
    'vegetation',
    'spalling',
    'corrosion',
    'efflorescence',
    'control_point',
)

S2DS_PALETTE = [
    [0, 0, 0],
    [255, 255, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
]

S2DS_COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (255, 255, 255): 1,
    (0, 255, 0): 2,
    (255, 0, 0): 3,
    (255, 255, 0): 4,
    (0, 255, 255): 5,
    (0, 0, 255): 6,
}


@DATASETS.register_module()
class S2DSDataset(CustomDataset):
    """S2DS dataset with RGB annotation support."""

    CLASSES = S2DS_CLASSES
    PALETTE = S2DS_PALETTE
    COLOR_TO_CLASS = S2DS_COLOR_TO_CLASS

    def __init__(self, **kwargs):
        kwargs.setdefault('img_suffix', '.png')
        kwargs.setdefault('seg_map_suffix', '_lab.png')
        kwargs.setdefault('color_to_class', self.COLOR_TO_CLASS)
        super(S2DSDataset, self).__init__(**kwargs)
