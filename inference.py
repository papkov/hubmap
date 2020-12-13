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
from torch.nn import Identity, Module
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules import dataset as D
from modules.model import get_segmentation_model
from modules.util import set_device_id

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
    use_cuda_merger: bool = True,
    save_raw: bool = False,  # works only with cuda merger now
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
        use_cuda_merger=use_cuda_merger,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.loader.valid_bs,
        num_workers=0,  # rasterio cannot be used with multiple workers
        shuffle=False,
        pin_memory=True,
    )

    # Allocate tiles
    tiles = None
    if not use_cuda_merger:
        # if use cuda merger, they were allocated in test_ds
        tiles = torch.zeros(
            len(test_ds),
            int(test_ds.tile_size / test_ds.scale_factor),
            int(test_ds.tile_size / test_ds.scale_factor),
            dtype=torch.int8,
        )

    # Test time augmentations
    tta = [
        Identity(),
        F.torch_rot90_cw,
        F.torch_rot180,
        F.torch_rot90_ccw,
        F.torch_transpose,
        F.torch_transpose_rot90_cw,
        F.torch_transpose_rot180,
        F.torch_transpose_rot90_ccw,
        # F.torch_fliplr,
        # F.torch_flipud,
    ]

    for i, (tiles_batch, coords_batch) in enumerate(tqdm(test_loader, desc="Predict")):
        for augmentation in tta:
            pred_batch = model(augmentation(tiles_batch).float().cuda())

            # Upscale if needed
            if test_ds.scale_factor != 1:
                pred_batch = interpolate(
                    pred_batch, scale_factor=1 / test_ds.scale_factor
                )

            if use_cuda_merger:
                # Integrate in allocated CUDA tensor
                # Integration adds weight to norm_mask, so no need to additionally normalize TTA
                test_ds.merger.integrate_batch(
                    batch=pred_batch, crop_coords=coords_batch
                )
            else:
                # Or add in tiles
                bs = len(pred_batch)
                tiles[i * bs : (i + 1) * bs] += (
                    pred_batch.cpu().squeeze() > threshold
                ).type(torch.int8)

    # Merge predicted tiles back and scale to the original image size
    if use_cuda_merger:
        test_ds.merger.merge_()
        if save_raw:
            path_merged = str(target_path / f"{test_ds.image_hash}.pt")
            torch.save(test_ds.merger.image, path_merged)

        test_ds.merger.threshold_(threshold)
        merged = test_ds.merger.image.cpu().numpy().squeeze().astype(np.uint8)

        # Crop to original size inplace
        merged = merged[
            test_ds.tiler.margin_top : test_ds.tiler.image_height
            + test_ds.tiler.margin_top,
            test_ds.tiler.margin_left : test_ds.tiler.image_width
            + test_ds.tiler.margin_left,
        ]

    else:
        # Leave major voting for now
        tiles = tiles // len(tta)
        merged = test_ds.tiler.merge(tiles.numpy()[..., None]).squeeze()

    # Save raw predictions for possible ensemble
    # path_merged = str(target_path / f"{test_ds.image_hash}.npy")
    # print(f"Save to {path_merged}")
    # np.save(path_merged, merged)

    # RLE encoding
    rle = {
        "id": test_ds.image_hash,
        "predicted": D.rle_encode_less_memory(merged),
    }

    del test_ds, test_loader, tiles, merged
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
    use_cuda_merger: bool = True,
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
            use_cuda_merger=use_cuda_merger,
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

    parser.add_argument(
        "--use_cuda_merger",
        "-c",
        default=True,
        type=bool,
    )
    # parser.add_argument("--ids", nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tile_size", help="Tile size", default=256, type=int)
    parser.add_argument("--tile_step", help="Tile step", default=224, type=int)
    parser.add_argument("--device", "-d", help="Device", default=2, type=int)

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
        classes=1,
    )

    # Load checkpoint
    path_ckpt = path / cfg.train.logdir / "checkpoints/best.pth"
    print(f"\nLoading checkpoint {str(path_ckpt)}")
    state_dict = torch.load(path_ckpt, map_location=torch.device("cpu"))[
        "model_state_dict"
    ]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = model.float()
    model.eval()
    del state_dict
    gc.collect()

    return inference_dir(
        test_path=args.test_path,
        cfg=cfg,
        model=model,
        target_path=target_path,
        scale_factor=args.scale_factor,
        tile_step=args.tile_step,
        tile_size=args.tile_size,
        use_cuda_merger=args.use_cuda_merger,
    )


if __name__ == "__main__":
    main()
