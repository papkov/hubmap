import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import matplotlib.path as mpath
import matplotlib.pyplot as plt
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
    use_ids: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> Tuple[List[Path], List[Path], List[str]]:
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

    return images, masks, unique_ids


@dataclass
class TrainDataset(Dataset):
    images: List[Path]
    masks: List[Path]
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
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


class SearchDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.images, self.masks, _ = get_file_paths(
            path="/gpfs/hpc/home/papkov/hubmap/data/hubmap-256x256/",
            # TODO remove hardcode
            use_ids=(0, 1, 2, 3, 4, 5, 6)
        )
        # Implement additional initialization logic if needed

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.images)

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.
        image = np.array(Image.open(self.images[index]).convert("RGB")).astype(np.uint8)
        mask = np.array(Image.open(self.masks[index]))[..., None].astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


@dataclass
class TiffFile(Dataset):
    path: Union[Path, str]
    tile_size: Union[Tuple[int, int], int] = 1024
    tile_step: Union[Tuple[int, int], int] = 896
    scale_factor: float = 1
    random_crop: bool = False
    num_threads: Union[str, int] = "all_cpus"
    weight: str = "pyramid"
    anatomical_structure: Optional[List[Dict[str, Any]]] = None
    filter_crops: bool = True

    def __post_init__(self):
        # Get image hash name
        self.image_hash = self.path.split("/")[-1].split(".")[0]

        # Read anatomical structure
        if self.anatomical_structure is None:
            with open(
                str(self.path).replace(".tiff", "-anatomical-structure.json"), "r"
            ) as f:
                self.anatomical_structure = json.load(f)

        # Open TIFF image
        identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
        self.image = rasterio.open(
            self.path, transform=identity, num_threads=self.num_threads
        )

        # Set up tiler
        self.tiler = ImageSlicer(
            self.image.shape,
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self.weight,
        )

        # Check and prepare all the crops
        self.crops = []
        self.batch_crops = []
        self.within_any = []
        self.within_region = []
        for i, crop in enumerate(self.tiler.crops):
            (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1) = self.tiler.crop_no_pad(
                i, self.random_crop
            )
            within_any, within_region = self.is_within((iy0, iy1), (ix0, ix1))
            if self.filter_crops:
                if not within_any:
                    continue
            self.batch_crops.append(self.tiler.crops[i])
            self.crops.append(((ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1)))
            self.within_any.append(within_any)
            self.within_region.append(within_region)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1) = self.crops[i]
        # print((x0, x1, y0, y1), (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1))

        # Allocate tile
        tile_width, tile_height = self.tiler.tile_size
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

        ret = {
            "image": tile,
            "crop": self.batch_crops[i],
            "tile_crop": (ix0, ix1, iy0, iy1, tx0, tx1, ty0, ty1),
            "within_any": self.within_any[i],
        }
        ret.update(self.within_region[i])
        return ret

    def plot_structure(self):
        for region in self.anatomical_structure:
            coords = np.array(region["geometry"]["coordinates"][0])
            plt.plot(
                coords[:, 0],
                coords[:, 1],
                label=region["properties"]["classification"]["name"],
            )
        plt.gca().invert_yaxis()
        plt.legend()

    def is_within(self, y: Tuple[int, int], x: Tuple[int, int]):
        """
        Check if tile corners are within anatomical structure
        :params y: tuple (y0, y1)
        :params x: tuple (x0, x1)
        :return: (bool within_any, dict within_region)
        """
        points = list(product(x, y))
        paths = {}
        for i, region in enumerate(self.anatomical_structure):
            coords = np.array(region["geometry"]["coordinates"][0])
            # coords[:, 1] = self.tiler.image_height - coords[:, 1]  # turn around to align image
            region = region["properties"]["classification"]["name"]
            paths[f"{region}_{i}"] = mpath.Path(coords)
        # if any corner is within the structure, consider the whole tile within
        within_region = {
            region: np.any(path.contains_points(points))
            for region, path in paths.items()
        }
        within_any = np.any(list(within_region.values()))
        return within_any, within_region


@dataclass
class TestDataset(TiffFile):
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        super().__post_init__()
        # Transforms
        self.transforms = albu.Compose(
            [albu.Normalize(mean=self.mean, std=self.std), ToTensorV2()]
        )
        self.denormalize = Denormalize(mean=self.mean, std=self.std)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ret = super().__getitem__(i)
        ret["image"] = self.transforms(image=ret["image"])["image"]
        return ret

    def __len__(self) -> int:
        return len(self.crops)


def get_training_augmentations():
    """
    https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
    :return: albumentation.Compose() of selected augmentation
    """
    transforms = albu.Compose(
        [
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomRotate90(),
            albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
            albu.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=40
            ),
        ]
    )
    return transforms
    # return albu.Compose(
    #     [
    #
    #         albu.HorizontalFlip(),
    #         albu.VerticalFlip(),
    #         albu.RandomRotate90(),
    #         albu.ShiftScaleRotate(
    #             shift_limit=0.0625,
    #             scale_limit=0.2,
    #             rotate_limit=15,
    #             p=0.9,
    #             border_mode=cv2.BORDER_REFLECT,
    #         ),
    #         albu.OneOf(
    #             [
    #                 albu.OpticalDistortion(p=0.3),
    #                 albu.GridDistortion(p=0.1),
    #                 albu.IAAPiecewiseAffine(p=0.3),
    #             ],
    #             p=0.3,
    #         ),
    #         albu.OneOf(
    #             [
    #                 albu.HueSaturationValue(10, 15, 10),
    #                 albu.CLAHE(clip_limit=2),
    #                 albu.RandomBrightnessContrast(),
    #             ],
    #             p=0.3,
    #         ),
    #     ]
    # )


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
