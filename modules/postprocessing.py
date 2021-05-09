import numpy as np
import skimage.morphology
from tqdm.auto import tqdm
from typing import Optional


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

    def __init__(self, msk: np.ndarray, tile_size: int = 1024, step_size: int = 128, area_threshold: int = 7000):
        assert len(msk.shape) == 2
        assert msk.dtype in [np.uint8, np.uint16, bool]
        self.msk = msk
        self.tile_size = tile_size
        self.tile_step = step_size
        self.area_threshold = area_threshold
        self.bboxes = self.get_bboxes()

    def get_bboxes(self):
        bboxes = []
        for y in range(0, self.msk.shape[0], self.tile_step):
            for x in range(0, self.msk.shape[1], self.tile_step):
                ye = y + self.tile_size
                xe = x + self.tile_size
                if y < self.msk.shape[0] and x < self.msk.shape[1]:
                    bboxes.append((y, x, ye, xe))
        return np.array(bboxes)

    def _remove_small_objects(self, area_threshold: Optional[int] = None):
        assert area_threshold < self.tile_size ** 2
        if area_threshold is None:
            area_threshold = self.area_threshold

        for bbox in tqdm(self.bboxes):
            sy, sx, ey, ex = bbox
            cropped_tile = self.msk[sy:ey, sx:ex]

            if np.any(cropped_tile):
                labeled_tile = skimage.morphology.label(cropped_tile)

                for label in np.unique(labeled_tile):
                    if label == 0:
                        continue

                    label_mask = labeled_tile == label

                    area = np.sum(label_mask)
                    if area < area_threshold and not is_label_on_tile_edge(label_mask):
                        labeled_tile[label_mask] = 0
                        print(
                            f"Removed object at [{sy}:{ey}, {sx}:{ex}] with size {area}"
                        )

                # one tile is now processed and ready to be placed back
                self.msk[sy:ey, sx:ex] = (labeled_tile > 0).astype(np.uint8)

    def __call__(self, *args, **kwargs):
        return self._remove_small_objects(*args, **kwargs)
