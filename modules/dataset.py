from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import rasterio
import tifffile as tiff
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from skimage.transform import rescale, resize
from torch import Tensor as T
from torch.nn import Upsample
from torch.utils.data import Dataset
from tqdm.auto import tqdm

Array = np.ndarray


def read_tiff(path: str):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    image = rasterio.open(path, transform=identity).read()
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
class TestDataset(Dataset):
    path: Union[Path, str]
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    tile_size: Union[Tuple[int, int], int] = 256
    tile_step: Union[Tuple[int, int], int] = 64
    scale_factor: float = 1
    use_cuda_merger: bool = False

    def __post_init__(self):
        # Get image path
        self.image_hash = self.path.split("/")[-1].split(".")[0]

        # Read image
        self.image = read_tiff(self.path)
        print(f"Read image {self.image.shape}")

        if self.scale_factor != 1:
            self.image = rescale(
                self.image,
                scale=self.scale_factor,
                preserve_range=True,
                multichannel=True,
                anti_aliasing=False,
            )
            print(f"Scale image {self.scale_factor} {self.image.shape}")

        # Split to tiles
        self.tiler = ImageSlicer(
            self.image.shape, tile_size=self.tile_size, tile_step=self.tile_step
        )
        # You can do something with tiles here
        self.tiles = [tile for tile in self.tiler.split(self.image)]

        # CUDA merger (might take a lot of memory)
        if self.use_cuda_merger:
            self.merger = CudaTileMerger(
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
        tile = self.tiles[i]
        crop = self.tiler.crops[i]
        tile = self.transforms(image=tile)["image"]
        return tile, crop

    def __len__(self) -> int:
        return len(self.tiles)

    def integrate_batch(self, pred_batch, coords_batch):
        if self.merger is None:
            return
        self.merger.integrate_batch(pred_batch, coords_batch)

    def merge(
        self,
        tiles: Union[List[Array], Array],
        scale: bool = False,
    ):
        """
        Merges a list of output batches
        :param tiles:
        :param scale:
        :return:
        """
        merged = self.tiler.merge(tiles).squeeze()
        if scale:
            merged = resize(
                merged,
                self.image.shape[:2] if len(merged.shape) == 2 else self.image.shape,
                anti_aliasing=False,
                preserve_range=True,
            )
        return merged


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
