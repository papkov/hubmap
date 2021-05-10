import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class NormalizeByImage(ImageOnlyTransform):
    """Divide pixel values by 255 = 2**8 - 1, subtract image mean per channel and divide by std per channel.

    Args:
        by_channel, bool: whether to calculate mean and std per channel (array) or together (float)

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self, by_channel: bool = False, always_apply: bool = False, p: float = 1.0
    ):
        super(NormalizeByImage, self).__init__(always_apply, p)
        self.mean = 0
        self.std = 1
        self.by_channel = by_channel

    def apply(self, image, **params):
        if self.by_channel:
            self.mean = np.mean(image, axis=(0, 1), dtype=np.float32, keepdims=True)
            self.std = np.std(image, axis=(0, 1), dtype=np.float32, keepdims=True)
        else:
            self.mean = np.mean(image, dtype=np.float32)
            self.std = np.std(image, dtype=np.float32)

        denominator = np.reciprocal(self.std, dtype=np.float32)

        image = image.astype(np.float32)
        image -= self.mean
        image *= denominator
        return image

    def get_transform_init_args_names(self):
        return ("by_channel",)
