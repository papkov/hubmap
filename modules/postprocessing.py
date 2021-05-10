from typing import Optional

import numpy as np
import skimage.morphology
from tqdm.auto import tqdm


def is_label_on_tile_edge(single_label_tile: np.ndarray):
    res = [
        np.any(single_label_tile[0, :]),
        np.any(single_label_tile[:, 0]),
        np.any(single_label_tile[-1, :]),
        np.any(single_label_tile[:, -1]),
    ]
    return any(res)


class TileByTileFilter:
    """
    Class for postprocessing predicted binary mask tile-by-tile.
    Currently only support filtering out too small objects.
    A sliding window slides over the input image and when it sees an object
    inside the window that is 1) too small, 2) does not touch tile borders
    then it will remove it from the initial image.
    Usage:
        filter_obj = TileByTileFilter(big_msk, tile_size=512, step_size=50)
        filter_obj(area_threshold=160000)
    Params:
        msk: np.ndarray - input binary mask
        tile_size=512  - traversing tile size; (sz x sz) square
        step_size=50   - step size for traversing tile
    """

    def __init__(
        self,
        tile_size: int = 512,
        tile_step: int = 128,
        area_threshold: int = 10000,
    ):
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.area_threshold = area_threshold

    def get_bboxes(self, msk: np.ndarray):
        bboxes = []
        for y in range(0, msk.shape[0], self.tile_step):
            for x in range(0, msk.shape[1], self.tile_step):
                ye = y + self.tile_size
                xe = x + self.tile_size
                if y < msk.shape[0] and x < msk.shape[1]:
                    bboxes.append((y, x, ye, xe))
        return np.array(bboxes)

    def _remove_small_objects(
        self, msk: np.ndarray, area_threshold: Optional[int] = None
    ):
        if area_threshold is None:
            area_threshold = self.area_threshold
        assert area_threshold < self.tile_size ** 2

        bboxes = self.get_bboxes(msk)
        iterator = tqdm(bboxes)
        counter = 0
        for bbox in iterator:
            sy, sx, ey, ex = bbox
            cropped_tile = msk[sy:ey, sx:ex]

            if np.any(cropped_tile):
                labeled_tile = skimage.morphology.label(cropped_tile)

                for label in np.unique(labeled_tile):
                    if label == 0:
                        continue

                    label_mask = labeled_tile == label

                    area = np.sum(label_mask)
                    if area < area_threshold and not is_label_on_tile_edge(label_mask):
                        labeled_tile[label_mask] = 0
                        counter += 1
                        iterator.set_postfix_str(
                            f"Removed {counter} object at [{sy}:{ey}, {sx}:{ex}] with size {area}"
                        )

                # one tile is now processed and ready to be placed back
                msk[sy:ey, sx:ex] = (labeled_tile > 0).astype(np.uint8)

        return msk

    def __call__(self, msk: np.ndarray, *args, **kwargs):
        assert msk.dtype in [np.uint8, np.uint16, bool]
        assert len(msk.shape) == 2
        return self._remove_small_objects(msk, *args, **kwargs)
