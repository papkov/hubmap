from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import rasterio
import tifffile as tiff
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from skimage.transform import rescale, resize
from torch import Tensor as T
from torch.nn import Upsample
from torch.utils.data import Dataset
from tqdm.auto import tqdm

Array = np.ndarray


def read_tiff(path: str, scale_factor: float):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    with rasterio.open(path, transform=identity) as image:
        image = image.read(
            out_shape=(
                image.count,
                int(image.height * scale_factor),
                int(image.width * scale_factor),
            ),
            resampling=Resampling.bilinear,
        )
        if image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        return image


def rle_encode(im):
    """
    https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# New version
def rle_encode_less_memory(pixels):
    """
    https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    This simplified method requires first and last pixel to be zero
    """
    pixels = pixels.T.flatten()
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(256, 256)):
    """
    https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


class TileMerger(CudaTileMerger):
    def merge_(self) -> None:
        """
        Inplace version of CudaTileMerger.merge()
        Substitute self.image with self.image / self.norm_mask
        :return: None
        """
        self.image.div_(self.norm_mask)

    def threshold_(self, threshold: float = 0.5) -> None:
        """
        Inplace thresholding of self.image
        :return: None
        """
        self.image.gt_(threshold)
        self.image.type(torch.int8)


@dataclass
class TrainDataset(Dataset):
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    path: Union[Path, str] = "../data/hubmap-256x256/"
    use_ids: Tuple[int] = (0, 1, 2, 3, 4, 5, 6)
    transforms: Optional[Union[albu.BasicTransform, Any]] = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.unique_ids = sorted(
            set(
                str(p).split("/")[-1].split("_")[0]
                for p in (self.path / "train").iterdir()
            )
        )

        self.images = sorted(
            [
                p
                for i in self.use_ids
                for p in (self.path / "train").glob(f"{self.unique_ids[i]}_*.png")
            ]
        )
        self.masks = sorted(
            [
                p
                for i in self.use_ids
                for p in (self.path / "masks").glob(f"{self.unique_ids[i]}_*.png")
            ]
        )

        self.transforms = albu.Compose(
            [
                albu.PadIfNeeded(256, 256, border_mode=cv2.BORDER_REFLECT_101),
                self.transforms,
                albu.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

        self.denormalize = Denormalize(mean=self.mean, std=self.std)

        assert len(self.images) == len(self.masks)

    def __getitem__(self, i: int):
        image = np.array(Image.open(self.images[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks[i]))

        sample = self.transforms(image=image, mask=mask)
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


@dataclass
class TiffFile(Dataset):
    path: Union[Path, str]
    tile_size: Union[Tuple[int, int], int] = 256
    tile_step: Union[Tuple[int, int], int] = 224
    scale_factor: float = 1
    random_crop: bool = False
    num_threads: Union[str, int] = "all_cpus"

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
        )

    def __len__(self):
        return len(self.tiler.crops)

    def __getitem__(self, i: int) -> Array:
        crop = self.tiler.crops[i]

        x, y, tile_width, tile_height = crop
        if self.random_crop:
            x = np.random.randint(0, self.tiler.image_width)
            y = np.random.randint(0, self.tiler.image_height)

        # Get original coordinates with padding
        x0 = x - self.tiler.margin_left  # may be negative
        y0 = y - self.tiler.margin_top
        x1 = x0 + tile_width  # may overflow image size
        y1 = y0 + tile_height

        # Restrict coordinated by image size
        ix0 = max(x0, 0)
        iy0 = max(y0, 0)
        ix1 = min(x1, self.image.shape[1])
        iy1 = min(y1, self.image.shape[0])

        # Set shifts for the tile
        tx0 = ix0 - x0  # >= 0
        ty0 = iy0 - y0
        tx1 = tile_width + ix1 - x1  # <= tile_width
        ty1 = tile_height + iy1 - y1  # <= tile_height

        # print((x0, x1, y0, y1), (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1))

        # Allocate tile
        tile = np.zeros((tile_width, tile_height, self.image.count), dtype=np.uint8)

        # Read window
        possible_retries = 3
        for retry in range(possible_retries):
            try:
                window = self.image.read(
                    [1, 2, 3],  # read all three channels
                    window=Window.from_slices((iy0, iy1), (ix0, ix1)),
                )
                # Reshape if necessary
                if window.shape[-1] != 3:
                    window = np.moveaxis(window, 0, -1)

                # Map read image to the tile
                tile[ty0:ty1, tx0:tx1] = window
                break

            except RasterioIOError as e:
                # TODO do nothing now, fix later
                print(f"RasterioIOError at index {i}, retry {retry}:", e)

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
    use_cuda_merger: bool = False

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

        # CUDA merger (might take a lot of memory)
        if self.use_cuda_merger:
            self.merger = TileMerger(
                self.tiler.target_shape, channels=1, weight=self.tiler.weight
            )
        else:
            self.merger = None

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

    def integrate_batch(self, pred_batch, coords_batch):
        if self.merger is None:
            return
        self.merger.integrate_batch(pred_batch, coords_batch)


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
