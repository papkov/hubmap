import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from skimage.transform import resize
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules import dataset as D
from modules.model import get_segmentation_model
from modules.util import set_device_id

PathT = Union[Path, str]


def inference_one(
    image_path: PathT,
    target_path: PathT,
    cfg: DictConfig,
    model: Module,
    scale_factor: float,
    tile_size: int,
    tile_step: int,
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
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.loader.valid_bs,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Predict tiles
    tiles = []
    for tiles_batch, coords_batch in tqdm(test_loader, desc="Predict"):
        tiles_batch = tiles_batch.float().cuda()
        pred_batch = model(tiles_batch).detach().cpu().numpy().squeeze()
        tiles.append(pred_batch)
    tiles = np.concatenate(tiles)
    print(tiles.shape)

    # Merge predicted tiles back and scale to the original image size
    merged = test_ds.merge(tiles[..., None], scale=True).squeeze()

    # Save raw predictions for possible ensembling
    path_merged = str(target_path / f"{test_ds.image_hash}.npy")
    print(f"Save to {path_merged}")
    np.save(path_merged, merged)

    # RLE encoding
    rle = {
        "id": test_ds.image_hash,
        # TODO select threshold
        "predicted": D.rle_encode((merged > 0.5).astype(np.uint8)),
    }

    del test_ds
    del tiles
    del merged
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
):
    test_path = Path(test_path)
    target_path = Path(target_path)

    images = sorted(list(test_path.glob("*.tiff")))
    rle_encodings = []
    for i, image_path in enumerate(images):
        print(f"\nPredict image {image_path.as_posix()}")
        rle_encodings.append(
            inference_one(
                image_path=image_path,
                target_path=target_path,
                cfg=cfg,
                model=model,
                scale_factor=scale_factor,
                tile_size=tile_size,
                tile_step=tile_step,
            )
        )

    rle_encodings = pd.DataFrame(rle_encodings)
    rle_encodings.to_csv(
        target_path / "submission.csv",
        # target_path / f"submission_{datetime.today().strftime('%Y%m%dT%H%M%S')}.csv",
        index=False,
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="HuBMAP inference")
    parser.add_argument("-p", "--path", help="Run path", type=str)
    parser.add_argument(
        "--test_path", help="Test data path", default="data/test", type=str
    )
    parser.add_argument(
        "-s", "--scale_factor", help="Scale factor", default=0.25, type=float
    )
    # parser.add_argument("--ids", nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tile_size", help="Tile size", default=256, type=int)
    parser.add_argument("--tile_step", help="Tile step", default=64, type=int)
    args = parser.parse_args()

    # Set working path
    print(f"Going to use {args.path}")
    path = Path(args.path)
    target_path = path / "inference"
    # TODO possible overwrite?
    target_path.mkdir(exist_ok=True)

    # Read config
    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = set_device_id(cfg.device)

    # Load model once
    model = get_segmentation_model(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=None,
        classes=1,
    )
    model = model.to(device)
    model.eval()

    # Load checkpoint
    path_ckpt = path / cfg.train.logdir / "checkpoints/best.pth"
    print(f"\nLoading checkpoint {str(path_ckpt)}")
    ckpt = torch.load(path_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])

    return inference_dir(
        test_path=args.test_path,
        cfg=cfg,
        model=model,
        target_path=target_path,
        scale_factor=args.scale_factor,
        tile_step=args.tile_step,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
