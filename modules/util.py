import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.io import DatasetReader


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
