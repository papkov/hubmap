import unittest

import numpy as np

from modules.util import rle_crop


class TestRLE(unittest.TestCase):
    def test_rle_crop(self):
        """
        Tests correctness of efficient RLE cropping

        rle = '1 1 13 6 21 2'
        mask =
        [[1, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]]


        :return:
        """
        tile = rle_crop("1 1 13 6 21 2", iy0=1, iy1=5, ix0=2, ix1=4, image_shape=(6, 5))
        expected = np.array([[1, 0], [1, 1], [1, 1], [1, 0]])

        self.assertEqual(tile.shape, expected.shape)
        self.assertTrue(np.all(tile == expected))


if __name__ == "__main__":
    unittest.main()
