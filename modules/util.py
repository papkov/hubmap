import os
from typing import Optional, Union

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling


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
