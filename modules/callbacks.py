from itertools import chain
from typing import TYPE_CHECKING, Dict, List, Tuple

import wandb
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder, CallbackScope
from catalyst.dl import IRunner, OptimizerCallback, SchedulerCallback
from catalyst.utils.dict import split_dict_to_subdicts
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class WandbLogger(Callback):
    """Logger callback, translates ``runner.*_metrics`` to Weights & Biases.
    Read about Weights & Biases here https://docs.wandb.com/

    Example:
        .. code-block:: python

            from catalyst import dl
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            class Projector(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.linear = nn.Linear(input_size, 1)

                def forward(self, X):
                    return self.linear(X).squeeze(-1)

            X = torch.rand(16, 10)
            y = torch.rand(X.shape[0])
            model = Projector(X.shape[1])
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=8)
            runner = dl.SupervisedRunner()

            runner.train(
                model=model,
                loaders={
                    "train": loader,
                    "valid": loader
                },
                criterion=nn.MSELoss(),
                optimizer=optim.Adam(model.parameters()),
                logdir="log_example",
                callbacks=[
                    dl.callbacks.WandbLogger(
                        project="wandb_logger_example"
                    )
                ],
                num_epochs=10
            )
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = False,
        log_on_epoch_end: bool = True,
        log: str = None,
        **logging_params,
    ):
        """
        Args:
            metric_names: list of metric names to log,
                if None - logs everything
            log_on_batch_end: logs per-batch metrics if set True
            log_on_epoch_end: logs per-epoch metrics if set True
            log: wandb.watch parameter. Can be "all", "gradients"
                or "parameters"
            **logging_params: any parameters of function `wandb.init`
                except `reinit` which is automatically set to `True`
                and `dir` which is set to `<logdir>`
        """
        super().__init__(
            order=CallbackOrder.logging,
            node=CallbackNode.master,
            scope=CallbackScope.experiment,
        )
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end
        self.log = log

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        if (self.log_on_batch_end and not self.log_on_epoch_end) or (
            not self.log_on_batch_end and self.log_on_epoch_end
        ):
            self.batch_log_suffix = ""
            self.epoch_log_suffix = ""
        else:
            self.batch_log_suffix = "_batch"
            self.epoch_log_suffix = "_epoch"

        self.logging_params = logging_params

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        mode: str,
        suffix="",
        commit=True,
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        def key_locate(key: str):
            """
            Wandb uses first symbol _ for it service purposes
            because of that fact, we can not send original metric names

            Args:
                key: metric name

            Returns:
                formatted metric name
            """
            if key.startswith("_"):
                return key[1:]
            return key

        metrics = {
            f"{key_locate(key)}/{mode}{suffix}": value
            for key, value in metrics.items()
            if key in metrics_to_log
        }
        wandb.log(metrics, step=step, commit=commit)

    def on_stage_start(self, runner: "IRunner"):
        """Initialize Weights & Biases."""
        wandb.init(**self.logging_params, reinit=True, dir=str(runner.logdir))
        if self.log is not None:
            wandb.watch(models=runner.model, criterion=runner.criterion, log=self.log)

    # def on_stage_end(self, runner: "IRunner"):
    #     """Finish logging to Weights & Biases."""
    #     wandb.join()

    def on_batch_end(self, runner: "IRunner"):
        """Translate batch metrics to Weights & Biases."""
        if self.log_on_batch_end:
            mode = runner.loader_key
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix=self.batch_log_suffix,
                commit=True,
            )

    def on_loader_end(self, runner: "IRunner"):
        """Translate loader metrics to Weights & Biases."""
        if self.log_on_epoch_end:
            mode = runner.loader_key
            metrics = runner.loader_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_epoch,
                mode=mode,
                suffix=self.epoch_log_suffix,
                commit=False,
            )

    def on_epoch_end(self, runner: "IRunner"):
        """Translate epoch metrics to Weights & Biases."""
        extra_mode = "_base"
        splitted_epoch_metrics = split_dict_to_subdicts(
            dct=runner.epoch_metrics,
            prefixes=list(runner.loaders.keys()),
            extra_key=extra_mode,
        )

        if self.log_on_epoch_end:
            if extra_mode in splitted_epoch_metrics.keys():
                # if we are using OptimizerCallback
                self._log_metrics(
                    metrics=splitted_epoch_metrics[extra_mode],
                    step=runner.global_epoch,
                    mode=extra_mode,
                    suffix=self.epoch_log_suffix,
                    commit=True,
                )


class ProgressiveResizingCallback(SchedulerCallback):
    def __init__(
        self,
        scales: Tuple[float, ...] = (0.125, 0.25, 0.5),
        epochs: Tuple[int, ...] = (15, 30, 40),
        reset_scheduler: bool = True,
        reduce_base_lr: int = 0.1,
        scheduler_key: str = None,
        mode: str = None,
        reduced_metric: str = None,
    ):
        """

        :param scales: scales to interpolate image
        :param epochs: end epochs to switch scaling, after epochs[-1] scale will be 1
        """
        super().__init__(
            scheduler_key=scheduler_key, mode=mode, reduced_metric=reduced_metric
        )
        # self.reset_scheduler = reset_scheduler
        # Create scaling schedule
        self.reduce_base_lr = reduce_base_lr
        self.scales = scales
        self.epochs = epochs
        self.scale_factor = scales[0]
        self.base_batch_size = None

    def on_stage_start(self, runner: "IRunner") -> None:
        """Stage start hook.

        Args:
            runner: current runner
        """
        super().on_stage_start(runner)
        self.base_batch_size = runner.batch_size

    def on_epoch_start(self, runner: "IRunner") -> None:
        """Event handler for epoch start.

        Args:
            runner: IRunner instance.
        """

        # Set appropriate scale factor
        for scale, epoch in zip(self.scales, self.epochs):
            if runner.epoch < epoch:
                self.scale_factor = scale
                break
        else:
            # no break meant that current epoch > any scheduled scale
            self.scale_factor = 1

        # Scale batch size in data loaders (easiest way is to redefine them)
        if self.scale_factor != 1:
            runner.batch_size = int(1 / self.scale_factor)
            for key, loader in runner.loaders:
                kwargs = {
                    k: v
                    for k, v in loader.__dict__
                    if not k.startswith("_") and key != "batch_size"
                }
                runner.loaders[key] = DataLoader(batch_size=runner.batch_size, **kwargs)

    def on_batch_start(self, runner: "IRunner"):
        """Event handler for batch start.
        Resize both training and validation batches according to the scaling schedule

        Args:
            runner: IRunner instance.
        """
        if self.scale_factor != 1:
            if isinstance(runner.input, dict):
                for k, v in runner.input:
                    runner.input[k] = interpolate(
                        v, scale_factor=self.scale_factor, mode="nearest"
                    )
            else:
                runner.input = interpolate(runner.input, scale_factor=self.scale_factor)
