import os
import random
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from torch import Tensor as T


def read_opened_rasterio(rasterio_image: DatasetReader, scale_factor: float = 1.0):
    image = rasterio_image.read(
        out_shape=(
            rasterio_image.count,
            int(rasterio_image.height * scale_factor),
            int(rasterio_image.width * scale_factor),
        ),
        resampling=Resampling.bilinear,
    )
    if image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    return image


def read_tiff(path: str, scale_factor: float = 1.0):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    with rasterio.open(path, transform=identity) as image:
        return read_opened_rasterio(image, scale_factor)


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


def rle_decode(mask_rle: Union[str, np.ndarray], shape=(256, 256)):
    """
    https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split() if isinstance(mask_rle, str) else mask_rle
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_crop(
    rle_full: Union[str, np.ndarray],
    iy0: int,
    iy1: int,
    ix0: int,
    ix1: int,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Crop a tile from RLE-encoded mask without allocating memory for the full-sized array
    :param rle_full: RLE-encoded mask as str of array
    :param iy0: coordinates to crop (inclusive)
    :param iy1:
    :param ix0:
    :param ix1:
    :param image_shape: shape of the full-size mask
    :return: binary mask of shape (iy1 - iy0, ix1 - ix0)
    """
    rle = copy(rle_full)
    if isinstance(rle, str):
        rle = np.array(rle.split(), dtype=int)
    if rle.ndim == 1:
        rle = rle.reshape(-1, 2)
    # shift to zero-base
    rle[:, 0] -= 1

    rows, cols = image_shape
    h = iy1 - iy0
    w = ix1 - ix0

    # filter out entries that are outside the region for sure
    start = rows * ix0 + iy0  # start of the tile column
    end = rows * ix1 + iy1  # end of the tile column
    subset = (rle.sum(1) > start) & (rle[:, 0] < end)
    if subset.sum() > 0:
        rle = rle[subset]

    rle_mask = []
    rle_ends = rle.sum(1)  # ends of encoding segments

    # iterate over columns (RLE uses column-major order)
    for i, x in enumerate(range(ix0, ix1)):
        start = rows * x + iy0  # start of the tile column
        end = rows * x + iy1  # end of the tile column
        subset = (rle_ends > start) & (rle[:, 0] < end)
        if subset.sum() > 0:
            rle_subset = rle[subset]
            rle_subset[:, 0] += -start
            # trim segments start if start < 0
            shift = -np.minimum(rle_subset[:, 0], 0)
            rle_subset[:, 1] -= shift
            rle_subset[:, 0] += shift + i * h  # also shift by column coordinate
            # trim segment end if c + l > end
            rle_subset[:, 1] -= np.maximum(rle_subset.sum(1) - (end - start + i * h), 0)
            assert np.all(rle_subset >= 0)
            rle_mask.append(rle_subset)

    if not rle_mask:
        return np.zeros((h, w), dtype=np.uint8)

    rle_mask = np.concatenate(rle_mask, 0)
    # shift to one-base
    rle_mask[:, 0] += 1
    mask = rle_decode(rle_mask.flatten(), (h, w))
    return mask


def set_device_id(device_id: Optional[Union[int, str]] = None) -> str:
    """
    Converts device id to cuda/cpu str for moving tensors
    :param device_id:
    :return: cuda or cpu
    """
    if device_id is not None and device_id != "cpu" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        return "cuda"
    else:
        return "cpu"


def polygon_area(x, y):
    """
    Shoelace formula to calculate polygon area
    https://stackoverflow.com/a/30408825
    :param x:
    :param y:
    :return:
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def set_seed(seed: int, device: str = "cuda"):
    print(f"Seed {seed} for device {device}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_plot(ax):
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def polygon2mask(anatomical_structure: List[Dict[str, Any]], shape: Tuple[int, int]):
    """
    Convert anatomical structure to the np.array of given shape
    :param anatomical_structure:
    :param shape:
    :return:
    """
    mask = np.zeros(shape).astype(np.uint8)
    for i, region in enumerate(anatomical_structure):
        for coords in region["geometry"]["coordinates"]:
            coords = np.array(coords, dtype=np.int32).squeeze()[None, ...]
            cv2.fillPoly(mask, coords, i + 1)
    return mask.astype(np.uint8)


def plot_batch(
    batch,
    nrows: int = 4,
    s: int = 2,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
):
    bs = len(batch["image"])
    denormalize = Denormalize(mean=mean, std=std)
    ncols = bs // nrows
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(s * ncols, s * nrows))
    for i, ax in enumerate(axes.flat):
        ax.imshow(denormalize(batch["image"][i]))
        ax.imshow(batch["mask"][i][0], alpha=0.3)
    clean_plot(axes)


@dataclass
class Denormalize:
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)

    def __call__(self, tensor: T, numpy: bool = True) -> Union[T, np.ndarray]:
        """
        :param tensor: image of size (C, H, W) to be normalized
        :return: normalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        if numpy:
            tensor = np.moveaxis(tensor.numpy(), 0, -1)
        return tensor
