from .builder import DATASETS
from .custom import CustomDataset


SEGMENTATION1641_CLASSES = (
    'background',
    'crack',
    'spall',
    'corrosion',
    'rebar',
)

SEGMENTATION1641_PALETTE = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
]

SEGMENTATION1641_COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (255, 255, 255): 1,
    (255, 0, 0): 2,
    (255, 255, 0): 3,
    (0, 0, 255): 4,
}


@DATASETS.register_module()
class Segmentation1641Dataset(CustomDataset):
    """segmentation1641 dataset with RGB annotation support."""

    CLASSES = SEGMENTATION1641_CLASSES
    PALETTE = SEGMENTATION1641_PALETTE
    COLOR_TO_CLASS = SEGMENTATION1641_COLOR_TO_CLASS

    def __init__(self, **kwargs):
        kwargs.setdefault('img_suffix', '.jpg')
        kwargs.setdefault('seg_map_suffix', '_lab.png')
        kwargs.setdefault('color_to_class', self.COLOR_TO_CLASS)
        super(Segmentation1641Dataset, self).__init__(**kwargs)
