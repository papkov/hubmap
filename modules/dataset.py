from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import rasterio
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_toolbelt.inference.tiles import ImageSlicer
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch import Tensor as T
from torch.utils.data import Dataset, SubsetRandomSampler

Array = np.ndarray
PathT = Union[Path, str]


def get_file_paths(
    path: PathT = "../data/hubmap-256x256/",
    use_ids: Tuple[int] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> Tuple[List[Path], List[Path]]:
    """
    Get lists of paths to training images and masks
    :param path: path to the data
    :param use_ids: ids to use
    :return:
    """
    path = Path(path)
    unique_ids = sorted(
        set(str(p).split("/")[-1].split("_")[0] for p in (path / "train").iterdir())
    )

    images = sorted(
        [p for i in use_ids for p in (path / "train").glob(f"{unique_ids[i]}_*.png")]
    )

    masks = sorted(
        [p for i in use_ids for p in (path / "masks").glob(f"{unique_ids[i]}_*.png")]
    )

    assert len(images) == len(masks)

    return images, masks


@dataclass
class TrainDataset(Dataset):
    images: List[Path]
    masks: List[Path]
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    transforms: Optional[Union[albu.BasicTransform, Any]] = None

    def __post_init__(self):

        assert len(self.images) == len(self.masks)

        self.base_transforms = albu.Compose(
            [
                albu.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

        self.applied_transforms = albu.Compose(
            [
                self.transforms,
                self.base_transforms,
            ]
        )

        self.denormalize = Denormalize(mean=self.mean, std=self.std)

    def __getitem__(self, i: int):
        image = np.array(Image.open(self.images[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks[i]))

        sample = self.applied_transforms(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        ret = {
            "i": i,
            "image_id": str(self.images[i]).split("/")[-1].split("_")[0],
            "image": image.float(),
            "mask": mask.float().unsqueeze(0),
        }

        return ret

    def __len__(self):
        return len(self.images)

    # def train(self):
    #     self.applied_transforms = albu.Compose(
    #         [
    #             self.transforms,
    #             self.base_transforms,
    #         ]
    #     )
    #
    # def eval(self):
    #     self.applied_transforms = self.base_transforms


@dataclass
class TiffFile(Dataset):
    path: Union[Path, str]
    tile_size: Union[Tuple[int, int], int] = 256
    tile_step: Union[Tuple[int, int], int] = 192
    scale_factor: float = 1
    random_crop: bool = False
    num_threads: Union[str, int] = "all_cpus"
    weight: str = "pyramid"

    def __post_init__(self):
        # Get image hash name
        self.image_hash = self.path.split("/")[-1].split(".")[0]

        # Open TIFF image
        identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
        self.image = rasterio.open(
            self.path, transform=identity, num_threads=self.num_threads
        )

        # Set up tiler
        self.tiler = ImageSlicer(
            self.image.shape,
            tile_size=int(self.tile_size / self.scale_factor),
            tile_step=int(self.tile_step / self.scale_factor),
            weight=self.weight,
        )

    def __len__(self):
        return len(self.tiler.crops)

    def __getitem__(self, i: int) -> Array:
        x, y, tile_width, tile_height = self.tiler.crops[i]
        (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1) = self.tiler.crop_no_pad(
            i, self.random_crop
        )
        # print((x0, x1, y0, y1), (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1))

        # Allocate tile
        tile = np.zeros((tile_width, tile_height, self.image.count), dtype=np.uint8)

        # Read window
        window = self.image.read(
            [1, 2, 3],  # read all three channels
            window=Window.from_slices((iy0, iy1), (ix0, ix1)),
        )
        # Reshape if necessary
        if window.shape[-1] != 3:
            window = np.moveaxis(window, 0, -1)

        # Map read image to the tile
        tile[ty0:ty1, tx0:tx1] = window

        # Scale the tile
        tile = cv2.resize(
            tile,
            (
                int(tile_width * self.scale_factor),
                int(tile_height * self.scale_factor),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return tile


@dataclass
class TestDataset(TiffFile):
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    cache_tiles: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.cache_tiles:
            # Read full-sized image
            image = self.image.read(
                out_shape=(
                    self.image.count,
                    int(self.image.height * self.scale_factor),
                    int(self.image.width * self.scale_factor),
                ),
                resampling=Resampling.bilinear,
            )
            if image.shape[-1] != 3:
                image = np.moveaxis(image, 0, -1)
            self.tiles = [tile for tile in self.tiles.split(image)]
        else:
            self.tiles = None

        # Transforms
        self.transforms = albu.Compose(
            [albu.Normalize(mean=self.mean, std=self.std), ToTensorV2()]
        )
        self.denormalize = Denormalize(mean=self.mean, std=self.std)

    def __getitem__(self, i: int) -> Tuple[T, Array]:
        crop = self.tiler.crops[i]
        if self.cache_tiles:
            tile = self.tiles[i]
        else:
            tile = super(TestDataset, self).__getitem__(i)

        tile = self.transforms(image=tile)["image"]
        return tile, crop

    def __len__(self) -> int:
        return len(self.tiler.crops)


def get_training_augmentations():
    """
    https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
    :return: albumentation.Compose() of selected augmentation
    """
    return albu.Compose(
        [
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomRotate90(),
            albu.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.9,
                border_mode=cv2.BORDER_REFLECT,
            ),
            albu.OneOf(
                [
                    albu.OpticalDistortion(p=0.3),
                    albu.GridDistortion(p=0.1),
                    albu.IAAPiecewiseAffine(p=0.3),
                ],
                p=0.3,
            ),
            albu.OneOf(
                [
                    albu.HueSaturationValue(10, 15, 10),
                    albu.CLAHE(clip_limit=2),
                    albu.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
        ]
    )


@dataclass
class Denormalize:
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)

    def __call__(self, tensor: T, numpy: bool = True):
        """
        :param tensor: image of size (C, H, W) to be normalized
        :return: normalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        if numpy:
            tensor = np.moveaxis(tensor.numpy(), 0, -1)
        return tensor
