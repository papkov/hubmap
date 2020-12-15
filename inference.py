import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_toolbelt.inference import functional as F
from pytorch_toolbelt.inference.tiles import TileMerger
from torch import sigmoid
from torch.nn import Identity, Module
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules import dataset as D
from modules.model import get_segmentation_model
from modules.util import rle_encode_less_memory, set_device_id

PathT = Union[Path, str]


@torch.no_grad()
def inference_one(
    image_path: PathT,
    target_path: PathT,
    cfg: DictConfig,
    model: Module,
    scale_factor: float,
    tile_size: int,
    tile_step: int,
    threshold: float = 0.5,
    save_raw: bool = False,
    tta_mode: int = 8,
    weight: str = "pyramid",
    device: str = "cuda",
):
    image_path = Path(image_path)
    target_path = Path(target_path)

    test_ds = D.TestDataset(
        image_path.as_posix(),
        mean=cfg.data.mean,
        std=cfg.data.std,
        scale_factor=scale_factor,
        tile_size=tile_size,
        tile_step=tile_step,
        weight=weight,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.loader.valid_bs,
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

    # Test time augmentations
    tta = [
        (F.torch_fliplr, F.torch_fliplr),
        (F.torch_flipud, F.torch_flipud),
        (F.torch_rot180, F.torch_rot180),
        (F.torch_rot90_cw, F.torch_rot90_ccw),
        (F.torch_rot90_ccw, F.torch_rot90_cw),
        (F.torch_transpose, F.torch_transpose),
        (F.torch_transpose_rot90_cw, F.torch_rot90_ccw_transpose),
        (F.torch_transpose_rot180, F.torch_rot180_transpose),
        (F.torch_transpose_rot90_ccw, F.torch_rot90_cw_transpose),
    ]
    if tta_mode == 8:
        tta = tta[2:]
    elif tta_mode == 4:
        tta = tta[:3]
    elif tta_mode == 0:
        tta = []
    else:
        raise ValueError

    for i, batch in enumerate(tqdm(test_loader, desc="Predict")):
        tiles_batch = batch["image"]

        # Allocate zeros
        bs = tiles_batch.shape[0]
        pred_batch = (
            torch.empty(bs, 1, tile_size, tile_size)
            .fill_(merger.default_value)
            .float()
            .to(device)
        )

        # Predict only non-empty batches
        if not np.any(batch["within_any"].numpy()):
            continue

        to_predict = torch.tensor(np.argwhere(batch["within_any"].numpy()))[..., 0]

        # Predict only tiles within structure
        tiles_batch = tiles_batch[to_predict].float().to(device)
        pred_batch[to_predict] = model(tiles_batch)

        # Run TTA, if any
        for aug, deaug in tta:
            # TODO decorator?
            pred_batch[to_predict] += deaug(model(aug(tiles_batch)))
        # Mean reduce
        pred_batch[to_predict] /= len(tta) + 1

        # Upscale if needed
        if test_ds.scale_factor != 1:
            pred_batch = interpolate(pred_batch, scale_factor=1 / test_ds.scale_factor)

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
    merged = merger.image.cpu().numpy().squeeze().astype(np.uint8)

    # Crop to original size inplace
    merged = merged[
        test_ds.tiler.margin_top : test_ds.tiler.image_height
        + test_ds.tiler.margin_top,
        test_ds.tiler.margin_left : test_ds.tiler.image_width
        + test_ds.tiler.margin_left,
    ]

    # RLE encoding
    rle = {
        "id": test_ds.image_hash,
        "predicted": rle_encode_less_memory(merged),
    }

    del test_ds, test_loader, merged
    gc.collect()
    torch.cuda.empty_cache()

    return rle


def inference_dir(
    test_path: PathT,
    target_path: PathT,
    cfg: DictConfig,
    model: Module,
    scale_factor: float,
    tile_size: int,
    tile_step: int,
    save_raw: bool = False,
    tta_mode: int = 8,
    threshold: float = 0.5,
    weight: str = "pyramid",
    device: str = "gpu",
):
    test_path = Path(test_path)
    target_path = Path(target_path)

    images = sorted(list(test_path.glob("*.tiff")))
    rle_encodings = []
    for i, image_path in enumerate(images):
        print(f"\nPredict image {image_path.as_posix()}")
        rle = inference_one(
            image_path=image_path,
            target_path=target_path,
            cfg=cfg,
            model=model,
            scale_factor=scale_factor,
            tile_size=tile_size,
            tile_step=tile_step,
            save_raw=save_raw,
            tta_mode=tta_mode,
            weight=weight,
            threshold=threshold,
            device=device,
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
        default="runs/unetplusplus-resnext50_32x4d/2020-12-09/00-28-06/",
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
    parser.add_argument("--tta", default=4, choices=[0, 4, 8], type=int)

    # parser.add_argument("--ids", nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tile_size", help="Tile size", default=256, type=int)
    parser.add_argument("--tile_step", help="Tile step", default=192, type=int)
    parser.add_argument("--device", "-d", help="Device", default=2, type=int)
    parser.add_argument(
        "--threshold",
        "-t",
        help="Threshold for prediction binarization",
        default=0.5,
        type=float,
    )
    parser.add_argument("--save_raw", action="store_true")

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
    )


if __name__ == "__main__":
    main()
