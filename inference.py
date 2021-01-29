import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_toolbelt.inference.tiles import TileMerger
from skimage.morphology import remove_small_holes, remove_small_objects
from torch.nn import Module
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules import dataset as D
from modules.model import get_segmentation_model
from modules.tta import get_tta
from modules.util import rle_encode_less_memory, set_device_id

PathT = Union[Path, str]


@torch.no_grad()
def inference_one(
    image_path: PathT,
    target_path: PathT,
    cfg: DictConfig,
    model: Union[Module, List[Module]],
    scale_factor: float,
    tile_size: int,
    tile_step: int,
    batch_size: int = 1,
    threshold: float = 0.5,
    interpolate_mode: str = "bicubic",
    filter_crops: Optional[str] = None,
    save_raw: bool = False,
    tta_mode: Optional[str] = None,
    weight: str = "pyramid",
    device: str = "cuda",
    stats: Optional[str] = None,
    recompute_stats: bool = False,
    refine: bool = False,
    postprocess: bool = False,
    roll: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[Dict[str, Any], Tuple[int, int]]:
    """

    :param image_path:
    :param target_path:
    :param cfg:
    :param model:
    :param scale_factor:
    :param tile_size:
    :param tile_step:
    :param batch_size:
    :param threshold:
    :param filter_crops:
    :param save_raw:
    :param tta_mode:
    :param weight:
    :param device:
    :param stats:
    :param recompute_stats: if calculate full-image statistics anew
    :param refine:
    :param roll: some gt masks might be shifted {"afa5e8098": (-40, -24)}
    :return:
    """
    image_path = Path(image_path)
    target_path = Path(target_path)
    image_id = str(image_path).split("/")[-1].split(".")[0]

    mean, std = cfg.data.mean, cfg.data.std
    if stats is not None:
        try:
            with open(stats, "r") as f:
                print(f"Use stats from {stats} for id {image_id}")
                stats: Dict[str, Dict[str, List[float]]] = json.load(f)
        except:
            stats = {}  # leads to recompute_stats = True

        # If stats were used in training (present in config), but current image is not there, recompute
        if image_id not in stats.keys():
            print(f"Did not find id {image_id} in stats, recomputing")
            recompute_stats = True
        # Else, substitute config mean
        else:
            mean = stats[image_id]["mean"]
            std = stats[image_id]["std"]

    test_ds = D.TestDataset(
        image_path.as_posix(),
        mean=mean,
        std=std,
        recompute_stats=recompute_stats,
        scale_factor=scale_factor,
        tile_size=tile_size,
        tile_step=tile_step,
        weight=weight,
        filter_crops=filter_crops,
    )

    print(test_ds.mean, test_ds.std)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,  # just do it one by one
        num_workers=0,  # rasterio cannot be used with multiple workers
        shuffle=False,
        pin_memory=True,
    )

    # Allocate tiles
    merger = TileMerger(
        test_ds.tiler.target_shape,
        channels=1,
        weight=test_ds.tiler.weight,
        device=device,
    )
    print("TileMerger initialized")

    # Wrap model with TTA
    if tta_mode is not None:
        try:
            model = get_tta(tta_mode, model)
            print(f"Apply TTA {tta_mode}")
        except ValueError:
            print("Do not apply TTA")

    iterator = tqdm(test_loader if batch_size > 1 else test_ds, desc="Predict")
    for i, batch in enumerate(iterator):
        # Iterate over dataset, hence need to unsqueeze batch dim
        if batch_size == 1:
            batch["image"] = batch["image"][None, ...]
            batch["crop"] = batch["crop"][None, ...]

        pred_batch = model(batch["image"].float().to(device))
        iterator.set_postfix(
            {
                "in_shape": tuple(batch["image"].shape),
                "out_shape": tuple(pred_batch.shape),
            }
        )

        # Upscale if needed
        if test_ds.scale_factor != 1:
            # TODO align_corners=True seems a bit better on val=7, check
            pred_batch = interpolate(
                pred_batch,
                scale_factor=1 / test_ds.scale_factor,
                mode=interpolate_mode,
                align_corners=True,
            )

        merger.integrate_batch(batch=pred_batch, crop_coords=batch["crop"])

    # Merge predicted tiles back
    merger.merge_()
    if save_raw:
        path_merged = str(target_path / f"{test_ds.image_hash}.pt")
        path_merged_norm_mask = str(target_path / f"{test_ds.image_hash}_norm_map.pt")
        print(f"Save to {path_merged}")
        torch.save(merger.image, path_merged)
        torch.save(merger.norm_mask, path_merged_norm_mask)

    merger.threshold_(threshold)
    merged = merger.image.cpu().numpy().squeeze().astype(bool)

    if postprocess:
        # TODO investigate
        min_glomerulus_area = (
            16384  # 23480 for val 7, 16972 for val 5 (remove_small_objects makes worse)
        )
        merged = remove_small_objects(
            merged, min_size=min_glomerulus_area, in_place=True
        )
        merged = remove_small_holes(
            merged, area_threshold=min_glomerulus_area, in_place=True
        )

    if roll is not None:
        if image_id in roll.keys():
            merged = np.roll(merged, roll[image_id], axis=(0, 1))

    # Crop to original size inplace
    merged = merged[
        test_ds.tiler.margin_start[0] : test_ds.tiler.image_shape[0]
        + test_ds.tiler.margin_start[0],
        test_ds.tiler.margin_start[1] : test_ds.tiler.image_shape[1]
        + test_ds.tiler.margin_start[1],
    ]

    # Check if predicted array is of the same shape as input
    assert test_ds.image.shape == merged.shape

    # RLE encoding
    rle = {
        "id": test_ds.image_hash,
        "predicted": rle_encode_less_memory(merged),
    }
    target_shape = merged.shape

    test_ds.close()
    del test_ds, test_loader, merged
    gc.collect()
    torch.cuda.empty_cache()

    return rle, target_shape


def inference_dir(
    test_path: PathT,
    target_path: PathT,
    cfg: DictConfig,
    model: Union[Module, List[Module]],
    scale_factor: float,
    tile_size: int = 1024,
    tile_step: int = 704,
    batch_size: int = 1,
    save_raw: bool = False,
    filter_crops: Optional[str] = None,
    tta_mode: Optional[str] = None,
    threshold: float = 0.5,
    postprocess: bool = False,
    interpolate_mode: str = "bicubic",
    weight: str = "pyramid",
    device: str = "gpu",
    stats: Optional[str] = None,
    roll: Optional[Dict[str, Tuple[int, int]]] = None,
):
    test_path = Path(test_path)
    target_path = Path(target_path)

    images = sorted(list(test_path.glob("*.tiff")))
    rle_encodings = []
    for i, image_path in enumerate(images):
        print(f"\nPredict image {image_path.as_posix()}")
        rle, shape = inference_one(
            image_path=image_path,
            target_path=target_path,
            cfg=cfg,
            model=model,
            scale_factor=scale_factor,
            tile_size=tile_size,
            tile_step=tile_step,
            interpolate_mode=interpolate_mode,
            batch_size=batch_size,
            save_raw=save_raw,
            tta_mode=tta_mode,
            weight=weight,
            threshold=threshold,
            postprocess=postprocess,
            device=device,
            filter_crops=filter_crops,
            roll=roll,
            stats=stats,
        )
        rle_encodings.append(rle)

    rle_encodings = pd.DataFrame(rle_encodings)
    rle_encodings.to_csv(
        target_path / "submission.csv",
        # target_path / f"submission_{datetime.today().strftime('%Y%m%dT%H%M%S')}.csv",
        index=False,
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="HuBMAP inference")
    parser.add_argument(
        "-p",
        "--path",
        help="Run path",
        type=str,
        # default="runs/unetplusplus-resnext50_32x4d/2020-12-09/00-28-06/",
        default="multirun/2020-12-19/20-16-34/0/",
    )
    parser.add_argument(
        "--test_path", help="Test data path", default="data/test", type=str
    )
    parser.add_argument(
        "-s", "--scale_factor", help="Scale factor", default=0.25, type=float
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        default=4,
        type=int,
    )
    parser.add_argument("--tta", default="d4", choices=["d1", "d2", "d4"], type=str)

    # parser.add_argument("--ids", nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tile_size", help="Tile size", default=1024, type=int)
    parser.add_argument("--tile_step", help="Tile step", default=704, type=int)
    parser.add_argument("--device", "-d", help="Device", default=1, type=int)
    parser.add_argument(
        "--threshold",
        "-t",
        help="Threshold for prediction binarization",
        default=0.5,
        type=float,
    )
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument(
        "--filter_crops", type=str, default="cortex", choices=["cortex", "tissue", None]
    )

    args = parser.parse_args()

    # Set working path
    print(f"Going to use {args.path}")
    path = Path(args.path)
    target_path = path / "inference"
    # TODO possible overwrite?
    target_path.mkdir(exist_ok=True)

    # Read config, overwrite if necessary
    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    cfg.loader.valid_bs = args.batch_size
    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = set_device_id(args.device)

    # Load model once
    model = get_segmentation_model(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=None,
        checkpoint_path=path / cfg.train.logdir / "checkpoints/best.pth",
        convert_bn=False
        if OmegaConf.is_missing(cfg.model, "convert_bn")
        else cfg.model.convert_bn,
        classes=1,
    )
    model = model.float()
    model = model.to(device)
    model.eval()

    return inference_dir(
        test_path=args.test_path,
        cfg=cfg,
        model=model,
        target_path=target_path,
        scale_factor=args.scale_factor,
        tile_step=args.tile_step,
        tile_size=args.tile_size,
        device=device,
        save_raw=args.save_raw,
        tta_mode=args.tta,
        threshold=args.threshold,
        filter_crops=args.filter_crops,
    )


if __name__ == "__main__":
    main()
