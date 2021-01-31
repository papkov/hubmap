import numpy as np
from albumentations import DualTransform
import cv2

class CopyPaste(DualTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, pool_size: int = 32):
        super(CopyPaste, self).__init__(always_apply=always_apply, p=p)
        self.pool_size = pool_size
        self.pool = []
        self.last_sampled = None

    def __call__(self, *args, force_apply=False, **kwargs):
        if len(self.pool) < self.pool_size:
            # populate pool
            sample = kwargs.copy()
            # rotate and flip randomly
            rot = np.random.choice(4)
            lr, ud = np.random.choice(2, size=2)
            for k, v in sample.items():
                sample[k] = np.rot90(v, rot)
                if lr:
                    sample[k] = np.fliplr(sample[k])
                if ud:
                    sample[k] = np.fliplr(sample[k])
            self.pool.append(sample)
            return kwargs
        else:
            if np.any(kwargs["mask"]):
                return super(CopyPaste, self).__call__(
                    *args, force_apply=force_apply, **kwargs
                )
            return kwargs

    def apply(self, img, **params):
        from_pool = np.random.choice(self.pool_size)
        self.last_sampled = self.pool.pop(from_pool)
        # TODO parametrize
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = cv2.dilate(self.last_sampled["mask"], kernel, iterations=1)
        mask = np.repeat(mask[..., None], 3, -1) > 0
        np.putmask(img, mask, self.last_sampled["image"])
        return img

    def apply_to_mask(self, img, **params):
        # Do not sample twice, `apply` is guaranteed to be called first
        img += self.last_sampled["mask"]
        return np.clip(img, 0, 1)
