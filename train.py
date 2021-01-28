import json
import os
from collections import OrderedDict
from pathlib import Path

import albumentations as albu
import hydra
import numpy as np
import pandas as pd
import torch
from adabelief_pytorch import AdaBelief
from catalyst import utils as cutils

# from catalyst.contrib.callbacks import WandbLogger
from catalyst.contrib.nn import BCEDiceLoss, DiceLoss, IoULoss, Lookahead, RAdam
from catalyst.dl import (
    CheckpointCallback,
    CriterionCallback,
    DiceCallback,
    EarlyStoppingCallback,
    IouCallback,
    MetricAggregationCallback,
    OptimizerCallback,
    SchedulerCallback,
    SupervisedRunner,
)
from catalyst.metrics.dice import dice
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

from inference import inference_one
from modules import dataset as D
from modules.callbacks import WandbLogger
from modules.loss import FocalTverskyLoss, LovaszLossBinary
from modules.model import (
    find_dice_threshold,
    get_optimizer,
    get_scheduler,
    get_segmentation_model,
)
from modules.util import rle_decode, set_device_id, set_seed


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
        if cfg.train.num_epochs == 0:
            cfg.data.scale_factor = resume_cfg.data.scale_factor
        OmegaConf.save(cfg, ".hydra/config.yaml")

    print(OmegaConf.to_yaml(cfg))

    device = set_device_id(cfg.device)
    set_seed(cfg.seed, device=device)

    # Augmentations
    if cfg.data.aug == "auto":
        transforms = albu.load(cwd / "autoalbument/autoconfig.json")
    else:
        transforms = D.get_training_augmentations()

    # Model
    print(
        f"Setup model {cfg.model.arch} {cfg.model.encoder_name} convert_bn={cfg.model.convert_bn}"
    )
    model = get_segmentation_model(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        classes=1,
        convert_bn=cfg.model.convert_bn,
        # decoder_attention_type="scse",  # TODO to config
    )
    model = model.to(device)
    model.train()
    print(model)

    # Optimization
    # Reduce LR for pretrained encoder
    layerwise_params = {
        "encoder*": dict(lr=cfg.optim.lr_encoder, weight_decay=cfg.optim.wd_encoder)
    }
    model_params = cutils.process_model_params(model, layerwise_params=layerwise_params)

    # Select optimizer
    optimizer = get_optimizer(
        name=cfg.optim.name,
        model_params=model_params,
        lr=cfg.optim.lr,
        wd=cfg.optim.wd,
        lookahead=cfg.optim.lookahead,
    )

    criterion = {
        "dice": DiceLoss(),
        # "dice": SoftDiceLoss(mode="binary", smooth=1e-7),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "lovasz": LovaszLossBinary(),
        "focal_tversky": FocalTverskyLoss(eps=1e-7, alpha=0.7, gamma=0.75),
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
                optimizer=optimizer
                if resume_cfg.optim.name == cfg.optim.name
                else None,
                criterion=criterion,
            )
        else:
            raise ValueError("Nothing to resume, checkpoint missing")

    # We could only want to validate resume, in this case skip training routine
    best_th = 0.5

    stats = None
    if cfg.data.stats:
        print(f"Use statistics from file: {cfg.data.stats}")
        stats = cwd / cfg.data.stats

    if cfg.train.num_epochs is not None:
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
            CriterionCallback(
                input_key="mask",
                prefix="loss_focal_tversky",
                criterion_key="focal_tversky",
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
                    "loss_focal_tversky": cfg.loss.focal_tversky,
                },
            ),
            # metrics
            DiceCallback(input_key="mask"),
            IouCallback(input_key="mask"),
            # gradient accumulation
            OptimizerCallback(accumulation_steps=cfg.optim.accumulate),
            # early stopping
            SchedulerCallback(reduced_metric="loss_dice", mode=cfg.scheduler.mode),
            EarlyStoppingCallback(**cfg.scheduler.early_stopping, minimize=False),
            # TODO WandbLogger works poorly with multistage right now
            WandbLogger(project=cfg.project, config=dict(cfg)),
            # CheckpointCallback(save_n_best=cfg.checkpoint.save_n_best),
        ]

        # Training
        runner = SupervisedRunner(
            device=device, input_key="image", input_target_key="mask"
        )

        # TODO Scheduler does not work now, every stage restarts from base lr
        scheduler_warm_restart = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[1, 2],
            gamma=10,
        )

        for i, (size, num_epochs) in enumerate(
            zip(cfg.data.sizes, cfg.train.num_epochs)
        ):
            scale = size / 1024
            print(
                f"Training stage {i}, scale {scale}, size {size}, epochs {num_epochs}"
            )

            # Datasets
            train_ds, valid_ds = D.get_train_valid_datasets_from_path(
                # path=(cwd / cfg.data.path),
                path=(cwd / f"data/hubmap-{size}x{size}/"),
                train_ids=cfg.data.train_ids,
                valid_ids=cfg.data.valid_ids,
                seed=cfg.seed,
                valid_split=cfg.data.valid_split,
                mean=cfg.data.mean,
                std=cfg.data.std,
                transforms=transforms,
                stats=stats,
            )

            train_bs = int(cfg.loader.train_bs / (scale ** 2))
            valid_bs = int(cfg.loader.valid_bs / (scale ** 2))
            print(
                f"train: {len(train_ds)}; bs {train_bs}",
                f"valid: {len(valid_ds)}, bs {valid_bs}",
            )

            # Data loaders
            data_loaders = D.get_data_loaders(
                train_ds=train_ds,
                valid_ds=valid_ds,
                train_bs=train_bs,
                valid_bs=valid_bs,
                num_workers=cfg.loader.num_workers,
            )

            # Select scheduler
            scheduler = get_scheduler(
                name=cfg.scheduler.type,
                optimizer=optimizer,
                num_epochs=num_epochs
                * (len(data_loaders["train"]) if cfg.scheduler.mode == "batch" else 1),
                eta_min=scheduler_warm_restart.get_last_lr()[0]
                / cfg.scheduler.eta_min_factor,
                plateau=cfg.scheduler.plateau,
            )

            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                callbacks=callbacks,
                logdir=cfg.train.logdir,
                loaders=data_loaders,
                num_epochs=num_epochs,
                verbose=True,
                main_metric=cfg.train.main_metric,
                load_best_on_end=True,
                minimize_metric=False,
                check=cfg.check,
                fp16=dict(amp=cfg.amp),
            )

            # Set new initial LR for optimizer after restart
            scheduler_warm_restart.step()
            print(f"New LR for warm restart {scheduler_warm_restart.get_last_lr()[0]}")

            # Find optimal threshold for dice score
            model.eval()
            best_th, dices = find_dice_threshold(model, data_loaders["valid"])
            print("Best dice threshold", best_th, np.max(dices[1]))
            np.save(f"dices_{size}.npy", dices)
    else:
        print("Validation only")
        # Datasets
        size = cfg.data.sizes[-1]
        train_ds, valid_ds = D.get_train_valid_datasets_from_path(
            # path=(cwd / cfg.data.path),
            path=(cwd / f"data/hubmap-{size}x{size}/"),
            train_ids=cfg.data.train_ids,
            valid_ids=cfg.data.valid_ids,
            seed=cfg.seed,
            valid_split=cfg.data.valid_split,
            mean=cfg.data.mean,
            std=cfg.data.std,
            transforms=transforms,
            stats=stats,
        )

        train_bs = int(cfg.loader.train_bs / (cfg.data.scale_factor ** 2))
        valid_bs = int(cfg.loader.valid_bs / (cfg.data.scale_factor ** 2))
        print(
            f"train: {len(train_ds)}; bs {train_bs}",
            f"valid: {len(valid_ds)}, bs {valid_bs}",
        )

        # Data loaders
        data_loaders = D.get_data_loaders(
            train_ds=train_ds,
            valid_ds=valid_ds,
            train_bs=train_bs,
            valid_bs=valid_bs,
            num_workers=cfg.loader.num_workers,
        )

        # Find optimal threshold for dice score
        model.eval()
        best_th, dices = find_dice_threshold(model, data_loaders["valid"])
        print("Best dice threshold", best_th, np.max(dices[1]))
        np.save(f"dices_val.npy", dices)

    #
    # # Load best checkpoint
    # checkpoint_path = Path(cfg.train.logdir) / "checkpoints/best.pth"
    # if checkpoint_path.exists():
    #     print(f"\nLoading checkpoint {str(checkpoint_path)}")
    #     state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
    #         "model_state_dict"
    #     ]
    #     model.load_state_dict(state_dict)
    #     del state_dict
    # model = model.to(device)
    # Load config for updating with threshold and metric
    # (otherwise loading do not work)
    cfg = OmegaConf.load(".hydra/config.yaml")
    cfg.threshold = float(best_th)

    # Evaluate on full-size image if valid_ids is non-empty
    df_train = pd.read_csv(cwd / "data/train.csv")
    df_train = {r["id"]: r["encoding"] for r in df_train.to_dict(orient="record")}
    dices = []
    unique_ids = sorted(
        set(
            str(p).split("/")[-1].split("_")[0]
            for p in (cwd / cfg.data.path / "train").iterdir()
        )
    )
    size = cfg.data.sizes[-1]
    scale = size / 1024
    for image_id in cfg.data.valid_ids:
        image_name = unique_ids[image_id]
        print(f"\nValidate for {image_name}")

        rle_pred, shape = inference_one(
            image_path=(cwd / f"data/train/{image_name}.tiff"),
            target_path=Path("."),
            cfg=cfg,
            model=model,
            scale_factor=scale,
            tile_size=cfg.data.tile_size,
            tile_step=cfg.data.tile_step,
            threshold=best_th,
            save_raw=True,
            tta_mode=None,
            weight="pyramid",
            device=device,
            filter_crops="tissue",
            stats=stats,
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
