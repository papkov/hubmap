from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch import Tensor as T
from torch.utils.data import Dataset


def rle2mask(mask_rle, shape):
    """
    https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    :param mask_rle: run-length as string formated (start length)
    :param shape: (width,height) of array to return
    :return: numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


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
                albu.Normalize(),
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


def get_training_augmentations():
    return albu.Compose(
        [
            albu.OneOf(
                [
                    albu.ShiftScaleRotate(
                        shift_limit=0, scale_limit=0, rotate_limit=10
                    ),
                    albu.OpticalDistortion(),
                    albu.NoOp(p=0.6),
                ]
            ),
            albu.OneOf(
                [
                    albu.CLAHE(),
                    albu.RandomBrightnessContrast(),
                    albu.RandomGamma(),
                    albu.HueSaturationValue(),
                    albu.NoOp(),
                ]
            ),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
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
