import json
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as albu
import cv2
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_toolbelt.inference.tiles import ImageSlicer
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from torch import Tensor as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from modules import util
from modules.augmentation import CopyPaste

Array = np.ndarray
PathT = Union[Path, str]


@dataclass
class TrainDataset(Dataset):
    images: List[Path]
    masks: List[Path]
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    stats_file: Optional[str] = None
    transforms: Optional[Union[albu.BasicTransform, Any]] = None

    def __post_init__(self):
        self.stats = None
        if self.stats_file:
            with open(self.stats_file, "r") as f:
                self.stats = json.load(f)

        assert len(self.images) == len(self.masks)
        self.to_tensor = ToTensorV2()
        self.dmean = copy(self.mean)
        self.dstd = copy(self.std)
        self.denormalize = Denormalize(mean=self.mean, std=self.std)

    def __getitem__(self, i: int):
        image_id = str(self.images[i]).split("/")[-1].split("_")[0]

        sample = dict(
            image=np.array(Image.open(self.images[i]).convert("RGB")),
            mask=np.array(Image.open(self.masks[i])),
        )
        if self.transforms is not None:
            sample = self.transforms(**sample)

        if self.stats is not None and image_id in self.stats:
            self.mean, self.std = self.stats[image_id]
        else:
            self.mean, self.std = copy(self.dmean), copy(self.dstd)

        sample["image"] = albu.normalize(sample["image"], self.mean, self.std)
        sample = self.to_tensor(**sample)

        ret = {
            "i": i,
            "image_id": image_id,
            "image": sample["image"].float(),
            "mask": sample["mask"].float().unsqueeze(0),
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
            use_ids=(0, 1, 2, 3, 4, 5, 6),
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
    filter_crops: Optional[str] = None  # "cortex" | "tissue"
    padding_mode: str = "constant"
    rle_mask: Optional[Union[str, Array]] = None

    def __post_init__(self):
        # Get image hash name
        self.image_hash = str(self.path).split("/")[-1].split(".")[0]

        # Read anatomical structure
        self.regions = []
        self.cortex = False
        if self.anatomical_structure is None and self.filter_crops is not None:
            try:
                with open(
                    str(self.path).replace(".tiff", "-anatomical-structure.json"), "r"
                ) as f:
                    self.anatomical_structure = json.load(f)
                    self.regions = [
                        r["properties"]["classification"]["name"]
                        for r in self.anatomical_structure
                    ]
                    # Check if cortex region is present
                    self.cortex = np.any(["cortex" in r.lower() for r in self.regions])
            except:
                print("Anatomical structure was not found, do not filter crops")
                self.filter_crops = None
                self.anatomical_structure = None

        # Open TIFF image
        self.image = rasterio.open(
            self.path,
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),  # identity
            num_threads=self.num_threads,
        )
        print("Image shape:", self.image.shape)

        # Set up tiler
        self.tiler = ImageSlicer(
            self.image.shape + (3,),  # add channel dim
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self.weight,
        )
        print("Slicer set up")

        # Check and prepare all the crops
        self.crops = []
        self.batch_crops = []
        self.within_any = []
        self.within_region = []
        for i, crop in enumerate(self.tiler.crops):
            ic, tc, crop, within_any, within_region, use_crop = self.process_tiler_crop(
                crop if not self.random_crop else None
            )
            if use_crop:
                self.batch_crops.append(crop)
                self.crops.append((ic, tc))
                self.within_any.append(within_any)
                self.within_region.append(within_region)

        # Process RLE mask for convenience
        if self.rle_mask is not None and isinstance(self.rle_mask, str):
            # reshape to (N, 2) with columns (coordinate, length)
            self.rle_mask = np.array(self.rle_mask.split(), dtype=int).reshape(-1, 2)

        print("Dataset initialized")

    def read_rle_crop(
        self, ix0: int, ix1: int, iy0: int, iy1: int
    ) -> Union[Array, None]:
        """
        Extracts a piece of RLE mask by coordinates and converts it to binary mask
        :param ix0: crop coordinates in original image
        :param ix1:
        :param iy0:
        :param iy1:
        :return: binary np.array of tile size or None if no mask
        """
        if self.rle_mask is None:
            return None
        return util.rle_crop(
            self.rle_mask,
            ix0=ix0,
            ix1=ix1,
            iy0=iy0,
            iy1=iy1,
            image_shape=self.image.shape,
        )

    def process_tiler_crop(
        self, crop: Optional[List[int]] = None
    ) -> Tuple[
        Tuple[int, ...], Tuple[int, ...], List[int], bool, Dict[str, bool], bool
    ]:
        """
        Processes tiler crop or generates a new one
        :param crop: crop coordinates from Tiler, generate random if None
        :return:
            ic, coordinates (ix0, ix1, iy0, iy1) in original image
            tc, coordinates (tx0, tx1, ty0, ty1) in tile
            crop, coordinates [x, y] of top left angle of the original crop, generated randomly
            within_any, bool
            within_region, Dict[str, bool]
            use_crop, bool, whether to use this crop (not filtered out)
        """

        if crop is None:
            # generate random crop
            # TODO move randomness to getitem
            crop = [
                np.random.randint(0, shape - ts - 1)
                for shape, ts in zip(self.image.shape, self.tiler.tile_size)
            ]
        (ic0, ic1), (tc0, tc1), crop = self.tiler.project_crop_to_tile(crop)
        (iy0, ix0), (iy1, ix1) = tuple(ic0), tuple(ic1)
        (ty0, tx0), (ty1, tx1) = tuple(tc0), tuple(tc1)

        # Check if a tile belongs to a region within the structure
        within_any, within_region, use_crop = self.check_use_crop(ix0, ix1, iy0, iy1)
        ic, tc = (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1)
        return ic, tc, crop, within_any, within_region, use_crop

    def check_use_crop(self, ix0: int, ix1: int, iy0: int, iy1: int):
        """
        Check if crop is within some region and should be used
        :param ix0: coordinates in original image
        :param ix1:
        :param iy0:
        :param iy1:
        :return:
            within_any, bool
            within_region, Dict[str, bool]
            use_crop, bool, whether to use this crop (not filtered out)
        """
        within_any = (True,)
        within_region = {"any": True}
        use_crop = True

        if self.filter_crops is not None:
            within_any, within_cortex, within_region = self.is_within(
                y=(iy0, iy1), x=(ix0, ix1)
            )

            if (
                # If cortex is present on the slide and crop is not in cortex, do not predict
                self.filter_crops == "cortex"
                and self.cortex
                and not within_cortex
                # If crop is not within any region, do not predict
            ) or not within_any:
                use_crop = False

        return within_any, within_region, use_crop

    def __len__(self) -> int:
        return len(self.crops)

    def close(self) -> None:
        self.image.close()

    def compute_image_stats(
        self, from_full_sized: bool = False
    ) -> Tuple[List[float], List[float]]:
        if from_full_sized:
            # here we need to read full file into memory, but numbers are accurate
            image = util.read_opened_rasterio(self.image, scale_factor=1)
            if self.anatomical_structure is not None:
                mask_region = (
                    util.polygon2mask(self.anatomical_structure, image.shape[:2])
                    .clip(0, 1)
                    .astype(bool)
                )
                image = image[mask_region]
            else:
                image = image.reshape(-1, 3)
            return list(image.mean(axis=0) / 255), list(image.std(axis=0) / 255)
        else:
            non_overlapping_tiler = ImageSlicer(
                self.image.shape + (3,),  # add channel dim
                tile_size=self.tile_size,
                tile_step=self.tile_size,  # not tile_step
                weight=self.weight,
            )

            # Read all the crops, approximate with mean
            means = []
            stds = []

            # sum_x = np.zeros(3)
            # sum_x2 = np.zeros(3)
            # n = 0

            for i, crop in enumerate(tqdm(non_overlapping_tiler.crops)):
                (
                    ((iy0, ix0), (iy1, ix1)),
                    ((ty0, tx0), (ty1, tx1)),
                    crop,
                ) = non_overlapping_tiler.project_crop_to_tile(crop)
                crop, mask = self.read_crop(
                    ((ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1)), pad_and_resize=False
                )

                within_any, within_region, use_crop = self.check_use_crop(
                    ix0, ix1, iy0, iy1
                )

                if use_crop:
                    # s = crop.sum(axis=(0, 1)) / 255
                    # sum_x += s
                    # sum_x2 += s ** 2
                    # n += np.product(crop.shape[:2])
                    means.append(crop.mean(axis=(0, 1)) / 255)
                    stds.append(crop.std(axis=(0, 1)) / 255)

            # mean = sum_x / n
            # std = np.sqrt((sum_x2 / n) - (mean * mean))
            # return list(mean), list(std)

            return list(np.mean(means, axis=0)), list(np.mean(stds, axis=0))

    def read_crop(
        self, crop: Tuple[Tuple[int, ...], Tuple[int, ...]], pad_and_resize: bool = True
    ) -> Tuple[Array, Array]:
        (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1) = crop

        # Allocate tile
        tile_width, tile_height = self.tiler.tile_size

        # Read window
        if self.image.count == 3:
            tile = self.image.read(
                [1, 2, 3],  # read all three channels
                window=Window.from_slices((iy0, iy1), (ix0, ix1)),
            ).astype(np.uint8)
        else:
            # https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/224883
            h = iy1 - iy0
            w = ix1 - ix0
            c = len(self.image.subdatasets)
            tile = np.zeros((h, w, c), dtype=np.uint8)
            for i, s in enumerate(self.image.subdatasets, 0):
                with rasterio.open(s) as layer:
                    tile[:, :, i] = layer.read(
                        1, window=Window.from_slices((iy0, iy1), (ix0, ix1))
                    ).astype(np.uint8)

        # Reshape if necessary
        if tile.shape[-1] != 3:
            tile = np.moveaxis(tile, 0, -1)

        # Read mask
        mask = self.read_rle_crop(ix0, ix1, iy0, iy1)

        if pad_and_resize:
            pad_width = [
                (ty0, self.tile_size - ty1),
                (tx0, self.tile_size - tx1),
                (0, 0),
            ]
            # Create tile by padding image slice to the tile size
            tile = np.pad(tile, pad_width=pad_width, mode=self.padding_mode)
            assert tile.shape == (self.tile_size, self.tile_size, 3)

            if mask is not None:
                mask = np.pad(mask, pad_width=pad_width[:-1], mode=self.padding_mode)
                assert mask.shape == (
                    self.tile_size,
                    self.tile_size,
                ), f"{mask.shape}, {self.tile_size}"

            # Scale the tile
            tile = cv2.resize(
                tile,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_AREA,
            )

            if mask is not None:
                mask = cv2.resize(
                    mask,
                    None,
                    fx=self.scale_factor,
                    fy=self.scale_factor,
                    interpolation=cv2.INTER_NEAREST,
                )
        return tile, mask

    def __getitem__(self, i: int) -> Dict[str, Any]:

        ic, tc = self.crops[i]
        crop = self.batch_crops[i]
        within_any = self.within_any[i]
        within_region = self.within_region[i]

        # update all the values for true randomness
        if self.random_crop:
            use_crop = False
            while not use_crop:
                (
                    ic,
                    tc,
                    crop,
                    within_any,
                    within_region,
                    use_crop,
                ) = self.process_tiler_crop()

        tile, mask = self.read_crop((ic, tc))
        (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1) = ic, tc

        ret = {
            "i": i,
            "image_id": self.image_hash,
            "image": tile,
            "mask": mask,
            "crop": crop,
            "tile_crop": (ix0, ix1, iy0, iy1, tx0, tx1, ty0, ty1),
            "within_any": within_any,
        }
        ret.update(within_region)
        return ret

    def plot_structure(self):
        if self.anatomical_structure is None:
            # do not proceed without anatomical structure
            print("Anatomical structure not found")
            return
        for region in self.anatomical_structure:
            for coords in region["geometry"]["coordinates"]:
                coords = np.array(coords, dtype=np.int32).squeeze()
                plt.plot(
                    coords[:, 0],
                    coords[:, 1],
                    label=region["properties"]["classification"]["name"],
                )
        plt.gca().invert_yaxis()
        plt.legend()

    def is_within(
        self,
        y: Tuple[int, int],
        x: Tuple[int, int],
        expansion: int = 0,
    ) -> Tuple[bool, bool, Dict[str, bool]]:
        """
        Check if tile corners are within anatomical structure
        :params y: tuple (y0, y1)
        :params x: tuple (x0, x1)
        :params expansion: how much to expand tile boundaries (allows including more tiles in the region)
        :return: (bool within_any, dict within_region)
        """
        if expansion != 0:
            # expand tile boundaries
            y = (min(0, y[0] - expansion), y[1] + expansion)
            x = (min(0, x[0] - expansion), x[1] + expansion)

        if self.anatomical_structure is None:
            # do not proceed without anatomical structure
            return True, True, {"any": True}
        points = list(product(x, y))
        paths = {}
        for i, region in enumerate(self.anatomical_structure):
            region_name = region["properties"]["classification"]["name"]
            for j, coords in enumerate(region["geometry"]["coordinates"]):
                coords = np.array(coords, dtype=np.int32).squeeze()
                paths[f"{region_name}_{i}_{j}"] = mpath.Path(coords)
        # if any corner is within the structure, consider the whole tile within
        within_region = {
            region: np.any(path.contains_points(points))
            for region, path in paths.items()
        }
        within_any = np.any(list(within_region.values()))
        within_cortex = np.any(
            [
                "cortex" in region.lower() and within
                for region, within in within_region.items()
            ]
        )
        return within_any, within_cortex, within_region


@dataclass
class TrainTiffDataset(Dataset):
    path_data: PathT = "data"
    drop_ids: Optional[Tuple[str]] = None
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    tile_size: Union[Tuple[int, int], int] = 1024
    tile_step: Union[Tuple[int, int], int] = 1024
    scale_factor: float = 1
    random_crop: bool = False
    num_threads: Union[str, int] = "all_cpus"
    filter_crops: Optional[str] = None  # "cortex" | "tissue"
    stats_file: Optional[str] = None
    transforms: Optional[Union[albu.BasicTransform, Any]] = None

    def __post_init__(self):
        self.path_data = Path(self.path_data)

        # read csv index for train and test (pseudolabels for test)
        self.index = []
        for split in ["train", "test"]:
            if (self.path_data / f"{split}.csv").exists():
                ind = pd.read_csv(self.path_data / f"{split}.csv")
                ind["split"] = split
                if self.drop_ids is not None:
                    ind = ind.loc[~ind["id"].isin(self.drop_ids), :]
                self.index.append(ind)
        self.index = pd.concat(self.index).reset_index()

        # concatenate tiff datasets
        self.dataset = ConcatDataset(
            [
                TiffFile(
                    path=self.path_data / f"{r['split']}/{r['id']}.tiff",
                    tile_size=self.tile_size,
                    tile_step=self.tile_step,
                    scale_factor=self.scale_factor,
                    random_crop=self.random_crop,
                    num_threads=self.num_threads,
                    filter_crops=self.filter_crops,
                    rle_mask=r["encoding"],
                )
                for i, r in self.index.iterrows()
            ]
        )

        # read stats if needed
        self.stats = None
        if self.stats_file:
            with open(self.stats_file, "r") as f:
                self.stats = json.load(f)

        self.to_tensor = ToTensorV2()
        self.dmean = copy(self.mean)
        self.dstd = copy(self.std)
        self.denormalize = Denormalize(mean=self.mean, std=self.std)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ret = self.dataset[i]
        sample = {k: v for k, v in ret.items() if k in ["image", "mask"]}
        if self.transforms is not None:
            sample = self.transforms(**sample)

        if self.stats is not None and ret["image_id"] in self.stats:
            self.mean, self.std = self.stats[ret["image_id"]]
        else:
            self.mean, self.std = copy(self.dmean), copy(self.dstd)

        sample["image"] = albu.normalize(sample["image"], self.mean, self.std)

        sample = self.to_tensor(**sample)
        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].float().unsqueeze(0)

        sample.update({k: v for k, v in ret.items() if k in ["i", "image_id"]})
        return sample


@dataclass
class TestDataset(TiffFile):
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    recompute_stats: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.recompute_stats:
            print("Recomputing image statistics")
            self.mean, self.std = self.compute_image_stats()
            print("Computed:", self.mean, self.std)

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
            # albu.RandomCrop(256, 256),
            # did not work
            # albu.MaskDropout(
            #     max_objects=1, image_fill_value=[160, 115, 173]  #  HuBMAP
            # ),
            # CopyPaste(pool_size=32),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomRotate90(),
            # albu.ShiftScaleRotate(scale_limit=(0.9, 1.1), rotate_limit=0),
            albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
            albu.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=50,
                val_shift_limit=50,  # was 40
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


def get_file_paths(
    path: PathT = "data/hubmap-256x256/",
    use_ids: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> Tuple[List[Path], List[Path], List[str]]:
    """
    Get lists of paths to training images and masks
    :param path: path to the data
    :param use_ids: ids to use
    :return:
    """
    path = Path(path)
    unique_ids = sorted(set(p.name.split("_")[0] for p in (path / "train").iterdir()))
    # print(unique_ids)

    images = sorted(
        [
            p
            for i in use_ids
            for p in (path / "train").glob(f"{unique_ids[i]}_*.png")
            if i < len(unique_ids)
        ]
    )

    masks = sorted(
        [
            p
            for i in use_ids
            for p in (path / "masks").glob(f"{unique_ids[i]}_*.png")
            if i < len(unique_ids)
        ]
    )

    assert len(images) == len(masks)

    return images, masks, unique_ids


def get_train_valid_path(
    path: Path,
    train_ids: Tuple[int],
    valid_ids: Optional[Tuple[int]] = None,
    seed: int = 56,
    valid_split: float = 0.1,
):
    train_images, train_masks, unique_ids = get_file_paths(path=path, use_ids=train_ids)
    if valid_ids is not None:
        print(f"Use ids {valid_ids} for validation")
        valid_images, valid_masks, _ = get_file_paths(path=path, use_ids=valid_ids)
    else:
        print("Use random stratified split")
        hashes = [str(img).split("/")[-1].split("_")[0] for img in train_images]
        train_images, valid_images, train_masks, valid_masks = train_test_split(
            train_images,
            train_masks,
            test_size=valid_split,
            random_state=seed,
            stratify=hashes,
        )
    return train_images, valid_images, train_masks, valid_masks


def get_train_valid_datasets(
    train_images: List[Path],
    train_masks: List[Path],
    valid_images: List[Path],
    valid_masks: List[Path],
    mean: Tuple[float],
    std: Tuple[float],
    scale: float = 0.5,
    path_tiff_data: Optional[str] = None,
    transforms: Optional[Union[albu.BasicTransform, Any]] = None,
    stats: Optional[str] = None,
) -> Tuple[TrainDataset, TrainDataset]:

    if path_tiff_data is None:
        train_ds = TrainDataset(
            images=train_images,
            masks=train_masks,
            mean=mean,
            std=std,
            transforms=transforms,
            stats_file=stats,
        )
    else:
        drop_ids = tuple(set([fp.name.split("_")[0] for fp in valid_images]))
        train_ds = TrainTiffDataset(
            path_data=path_tiff_data,
            mean=mean,
            std=std,
            transforms=transforms,
            stats_file=stats,
            scale_factor=scale,
            drop_ids=drop_ids,
            # TODO specify dict params
            filter_crops="tissue",
            random_crop=True,
        )
    valid_ds = TrainDataset(
        images=valid_images,
        masks=valid_masks,
        mean=mean,
        std=std,
        transforms=None,
        stats_file=stats,
    )
    return train_ds, valid_ds


def get_train_valid_datasets_from_path(
    path: Path,
    mean: Tuple[float],
    std: Tuple[float],
    train_ids: Tuple[int],
    valid_ids: Optional[Tuple[int]] = None,
    path_tiff_data: Optional[str] = None,
    scale: float = 0.5,
    seed: int = 56,
    valid_split: float = 0.1,
    transforms: Optional[Union[albu.BasicTransform, Any]] = None,
    stats: Optional[str] = None,
) -> Tuple[Union[TrainDataset, TrainTiffDataset], TrainDataset]:

    if valid_ids is None and path_tiff_data:
        print("Set path_tiff_data=None, because valid_ids not provided")
        path_tiff_data = None

    train_images, valid_images, train_masks, valid_masks = get_train_valid_path(
        path=path,
        train_ids=train_ids,
        valid_ids=valid_ids,
        seed=seed,
        valid_split=valid_split,
    )
    train_ds, valid_ds = get_train_valid_datasets(
        train_images=train_images,
        train_masks=train_masks,
        valid_images=valid_images,
        valid_masks=valid_masks,
        path_tiff_data=path_tiff_data,
        scale=scale,
        mean=mean,
        std=std,
        transforms=transforms,
        stats=stats,
    )
    return train_ds, valid_ds


def get_data_loaders(
    train_ds: Union[TrainDataset, TrainTiffDataset],
    valid_ds: TrainDataset,
    train_bs: int,
    valid_bs: int,
    num_workers: int,
) -> "OrderedDict[str, DataLoader]":
    return OrderedDict(
        train=DataLoader(
            train_ds,
            batch_size=train_bs,
            # set num_workers=0 for tiff because of rasterio
            num_workers=num_workers if isinstance(train_ds, TrainDataset) else 0,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        ),
        valid=DataLoader(
            valid_ds,
            batch_size=valid_bs,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        ),
    )
