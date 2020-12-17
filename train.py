import os
from collections import OrderedDict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from adabelief_pytorch import AdaBelief
from catalyst import utils as cutils
from catalyst.contrib.callbacks import WandbLogger
from catalyst.contrib.nn import DiceLoss, IoULoss, Lookahead, RAdam
from catalyst.dl import (
    CriterionCallback,
    DiceCallback,
    EarlyStoppingCallback,
    IouCallback,
    MetricAggregationCallback,
    SupervisedRunner,
)
from catalyst.metrics import dice
from catalyst.metrics.dice import dice
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_toolbelt.utils.random import set_manual_seed
from sklearn.model_selection import train_test_split
from torch import nn, optim, sigmoid
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from inference import inference_one
from modules import dataset as D
from modules.lovasz import LovaszLossBinary
from modules.model import batch_norm2en_resnet, get_segmentation_model
from modules.util import rle_decode, set_device_id


@torch.no_grad()
def find_dice_threshold(model: nn.Module, loader: DataLoader, device: str = "cuda"):
    dice_th_range = np.arange(0.1, 0.7, 0.01)
    masks = []
    preds = []
    dices = []
    for batch in tqdm(loader):
        preds.append(model(batch["image"].to(device)).cpu())
        masks.append(batch["mask"])
    masks = torch.cat(masks, dim=0)
    preds = torch.cat(preds, dim=0)
    for th in tqdm(dice_th_range):
        dices.append(dice(preds, masks, threshold=th).item())
        # dices.append(
        #     np.mean([dice(p, m, threshold=th).item() for p, m in zip(preds, masks)])
        # )
    best_th = dice_th_range[np.argmax(dices)]
    return best_th, (dice_th_range, dices)


@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):

    cwd = Path(get_original_cwd())

    # overwrite config if continue training from checkpoint
    resume_cfg = None
    if "resume" in cfg:
        cfg_path = cwd / cfg.resume / ".hydra/config.yaml"
        print(f"Continue from: {cfg.resume}")
        # Overwrite everything except device
        # TODO config merger (perhaps continue training with the same optimizer but other lrs?)
        resume_cfg = OmegaConf.load(cfg_path)
        cfg.model = resume_cfg.model
        OmegaConf.save(cfg, ".hydra/config.yaml")

    print(OmegaConf.to_yaml(cfg))

    set_manual_seed(cfg.seed)
    device = set_device_id(cfg.device)
    # wandb.init(project=cfg.project, config=cfg)

    train_images, train_masks, unique_ids = D.get_file_paths(
        path=(cwd / cfg.data.path), use_ids=cfg.data.train_ids
    )
    if cfg.data.valid_ids:
        print(f"Use ids {cfg.data.valid_ids} for validation")
        valid_images, valid_masks, _ = D.get_file_paths(
            path=(cwd / cfg.data.path), use_ids=cfg.data.valid_ids
        )
    else:
        print("Use random stratified split")
        hashes = [str(img).split("/")[-1].split("_")[0] for img in train_images]
        train_images, valid_images, train_masks, valid_masks = train_test_split(
            train_images,
            train_masks,
            test_size=cfg.data.valid_split,
            random_state=cfg.seed,
            stratify=hashes,
        )

    # Datasets
    train_ds = D.TrainDataset(
        images=train_images,
        masks=train_masks,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transforms=D.get_training_augmentations(),
    )
    valid_ds = D.TrainDataset(
        images=valid_images,
        masks=valid_masks,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transforms=None,
    )
    print("train:", len(train_ds), "valid:", len(valid_ds))

    # Data loaders
    data_loaders = OrderedDict(
        train=DataLoader(
            train_ds,
            batch_size=cfg.loader.train_bs,
            num_workers=cfg.loader.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        valid=DataLoader(
            valid_ds,
            batch_size=cfg.loader.valid_bs,
            num_workers=cfg.loader.num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    )

    # Model
    model = get_segmentation_model(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        classes=1,
        convert_bn=cfg.model.convert_bn,
    )
    model = model.to(device)
    model.train()

    # Optimization
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Reduce LR for pretrained encoder
    layerwise_params = {
        "encoder*": dict(lr=cfg.optim.lr_encoder, weight_decay=cfg.optim.wd_encoder)
    }
    model_params = cutils.process_model_params(model, layerwise_params=layerwise_params)

    # Select optimizer
    # TODO getter
    if cfg.optim.name == "radam":
        base_optimizer = RAdam(model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd)
    elif cfg.optim.name == "adabelief":
        base_optimizer = AdaBelief(
            model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
        )
    else:
        raise ValueError

    # Use lookahead
    if cfg.optim.lookahead:
        optimizer = Lookahead(base_optimizer)
    else:
        optimizer = base_optimizer

    # Select scheduler
    # TODO getter
    if cfg.scheduler.type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.num_epochs, eta_min=1e-8
        )
    elif cfg.scheduler.type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **cfg.scheduler.plateau
        )
    else:
        raise ValueError

    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "lovasz": LovaszLossBinary(),
    }

    # Load states if resuming training
    if "resume" in cfg:
        checkpoint_path = (
            cwd / cfg.resume / cfg.train.logdir / "checkpoints/best_full.pth"
        )
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint {str(checkpoint_path)}")
            checkpoint = cutils.load_checkpoint(checkpoint_path)
            cutils.unpack_checkpoint(
                checkpoint=checkpoint,
                model=model,
                optimizer=optimizer if resume_cfg.optim.name == cfg.optim.name else None,
                criterion=criterion
            )
        else:
            raise ValueError("Nothing to resume, checkpoint missing")

    # We could only want to validate
    if cfg.train.num_epochs > 0:
        callbacks = [
            # Each criterion is calculated separately.
            CriterionCallback(
                input_key="mask", prefix="loss_dice", criterion_key="dice"
            ),
            CriterionCallback(input_key="mask", prefix="loss_iou", criterion_key="iou"),
            CriterionCallback(input_key="mask", prefix="loss_bce", criterion_key="bce"),
            CriterionCallback(
                input_key="mask", prefix="loss_lovasz", criterion_key="lovasz"
            ),
            # And only then we aggregate everything into one loss.
            MetricAggregationCallback(
                prefix="loss",
                mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
                # because we want weighted sum, we need to add scale for each loss
                metrics={
                    "loss_dice": cfg.loss.dice,
                    "loss_iou": cfg.loss.iou,
                    "loss_bce": cfg.loss.bce,
                    "loss_lovasz": cfg.loss.lovasz,
                },
            ),
            # metrics
            DiceCallback(input_key="mask"),
            IouCallback(input_key="mask"),
            # early stopping
            EarlyStoppingCallback(**cfg.scheduler.early_stopping, minimize=False),
            WandbLogger(project=cfg.project, config=cfg),
        ]

        # Training
        runner = SupervisedRunner(
            device=device, input_key="image", input_target_key="mask"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            logdir=cfg.train.logdir,
            loaders=data_loaders,
            num_epochs=cfg.train.num_epochs,
            scheduler=scheduler,
            verbose=True,
            main_metric=cfg.train.main_metric,
            minimize_metric=False,
        )
    else:
        print("Validation only")

    # Load config for updating with threshold and metric
    # (otherwise loading do not work)
    cfg = OmegaConf.load(".hydra/config.yaml")

    # Load best checkpoint
    checkpoint_path = Path(cfg.train.logdir) / "checkpoints/best.pth"
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint {str(checkpoint_path)}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
        model.load_state_dict(state_dict)
        del state_dict
    model = model.to(device)
    model.eval()

    # Find optimal threshold for dice score
    best_th, dices = find_dice_threshold(model, data_loaders["valid"])
    print("Best dice threshold", best_th, np.max(dices[1]))
    np.save("dices.npy", dices)
    cfg.threshold = float(best_th)

    # Evaluate on full-size image if valid_ids is non-empty
    df_train = pd.read_csv(cwd / "data/train.csv")
    df_train = {r["id"]: r["encoding"] for r in df_train.to_dict(orient="record")}
    dices = []
    for image_id in cfg.data.valid_ids:
        image_name = unique_ids[image_id]
        print(f"\nValidate for {image_name}")

        rle_pred, shape = inference_one(
            image_path=(cwd / f"data/train/{image_name}.tiff"),
            target_path=Path("."),
            cfg=cfg,
            model=model,
            scale_factor=cfg.data.scale_factor,
            tile_size=cfg.data.tile_size,
            tile_step=cfg.data.tile_step,
            threshold=best_th,
            save_raw=True,
            tta_mode=0,
            weight="pyramid",
            device=device,
        )

        print("Predict", shape)
        pred = rle_decode(rle_pred["predicted"], shape)
        mask = rle_decode(df_train[image_name], shape)
        assert pred.shape == mask.shape, f"pred {pred.shape}, mask {mask.shape}"
        assert pred.shape == shape, f"pred {pred.shape}, expected {shape}"

        dices.append(
            dice(
                torch.from_numpy(pred).type(torch.uint8),
                torch.from_numpy(mask).type(torch.uint8),
                threshold=None,
                activation="none",
            )
        )
    print("Full image dice:", np.mean(dices))
    OmegaConf.save(cfg, ".hydra/config.yaml")
    return


if __name__ == "__main__":
    main()
