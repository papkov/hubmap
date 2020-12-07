import os
from collections import OrderedDict
from pathlib import Path

import hydra
import segmentation_models_pytorch as smp
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
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_toolbelt.utils.random import set_manual_seed
from torch import nn, optim
from torch.utils.data import DataLoader

from modules import dataset as D


@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_manual_seed(cfg.seed)
    if cfg.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
        device = "cuda"
    else:
        device = "cpu"

    cwd = Path(get_original_cwd())
    # wandb.init(project=cfg.project, config=cfg)

    # Datasets
    train_ds = D.TrainDataset(
        path=cwd / cfg.data.path,
        mean=cfg.data.mean,
        std=cfg.data.std,
        use_ids=cfg.data.train_ids,
        transforms=D.get_training_augmentations(),
    )
    valid_ds = D.TrainDataset(
        path=cwd / cfg.data.path,
        mean=cfg.data.mean,
        std=cfg.data.std,
        use_ids=cfg.data.valid_ids,
        transforms=None,
    )

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
    # TODO other architectures in cfg.model.arch
    model = smp.Linknet(encoder_name=cfg.model.encoder_name, classes=1)

    # Optimization
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Reduce LR for pretrained encoder
    layerwise_params = {
        "encoder*": dict(lr=cfg.optim.lr_encoder, weight_decay=cfg.optim.wd_encoder)
    }
    model_params = cutils.process_model_params(model, layerwise_params=layerwise_params)
    base_optimizer = RAdam(model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd)
    optimizer = Lookahead(base_optimizer)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler.plateau)

    criterion = {"dice": DiceLoss(), "iou": IoULoss(), "bce": nn.BCEWithLogitsLoss()}

    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(input_key="mask", prefix="loss_dice", criterion_key="dice"),
        CriterionCallback(input_key="mask", prefix="loss_iou", criterion_key="iou"),
        CriterionCallback(input_key="mask", prefix="loss_bce", criterion_key="bce"),
        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 1.0, "loss_iou": 0.5, "loss_bce": 0.5},
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
