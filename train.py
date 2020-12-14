import os
from collections import OrderedDict
from pathlib import Path

import hydra
import segmentation_models_pytorch as smp
from adabelief_pytorch import AdaBelief
from catalyst import utils as cutils
from catalyst.contrib.callbacks import WandbLogger
from catalyst.contrib.nn import DiceLoss, IoULoss, Lookahead, LovaszLossBinary, RAdam
from catalyst.dl import (
    CriterionCallback,
    DiceCallback,
    EarlyStoppingCallback,
    IouCallback,
    MetricAggregationCallback,
    SupervisedRunner,
)
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_toolbelt.utils.random import set_manual_seed
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from modules import dataset as D
from modules.model import batch_norm2en_resnet, get_segmentation_model
from modules.util import set_device_id


@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_manual_seed(cfg.seed)
    device = set_device_id(cfg.device)

    cwd = Path(get_original_cwd())
    # wandb.init(project=cfg.project, config=cfg)

    train_images, train_masks = D.get_file_paths(
        path=(cwd / cfg.data.path), use_ids=cfg.data.train_ids
    )
    if cfg.data.valid_ids:
        print(f"Use ids {cfg.data.valid_ids} for validation")
        valid_images, valid_masks = D.get_file_paths(
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
    )

    # Convert Batch Norm layers
    if cfg.model.convert_bn:
        print("Converting BatchNorm2d")
        batch_norm2en_resnet(model)

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

    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(input_key="mask", prefix="loss_dice", criterion_key="dice"),
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
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
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

    return


if __name__ == "__main__":
    main()
