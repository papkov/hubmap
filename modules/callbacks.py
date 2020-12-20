from catalyst.dl import SchedulerCallback, OptimizerCallback
from typing import Tuple
from itertools import chain
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader


class ProgressiveResizingCallback(SchedulerCallback):
    def __init__(self,
                 scales: Tuple[float, ...] = (0.25, 0.5),
                 epochs: Tuple[int, ...] = (20, 40),
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
        super().__init__(scheduler_key=scheduler_key, mode=mode, reduced_metric=reduced_metric)

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
            if runner.epoch < scale:
                self.scale_factor = scale
                break
        else:
            # no break meant that current epoch > any scheduled scale
            self.scale_factor = 1

        # Scale batch size in data loaders (easiest way is to redefine them)
        if self.scale_factor != 1:
            batch_size_scale = int(1 / self.scale_factor)
            for key, loader in runner.loaders:
                kwargs = {k: v for k, v in loader.__dict__ if not k.startswith("_") and key != "batch_size"}
                runner.loaders[key] = DataLoader(batch_size=batch_size_scale, **kwargs)

    def on_batch_start(self, runner: "IRunner"):
        """Event handler for batch start.
        Resize both training and validation batches according to the scaling schedule

        Args:
            runner: IRunner instance.
        """
        if self.scale_factor != 1:
            if isinstance(runner.input, dict):
                for k, v in runner.input:
                    runner.input[k] = interpolate(v, scale_factor=self.scale_factor, mode="nearest")
            else:
                runner.input = interpolate(runner.input, scale_factor=self.scale_factor)

