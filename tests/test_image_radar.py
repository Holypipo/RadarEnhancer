from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from config import DATA_DIR
from utils.image_radar import AEMET_PALETTE, image_to_dbz


ROOT_DIR = Path(__file__).resolve().parents[1]
GIF_20260515 = DATA_DIR / "aemet_nacional_20260515_1146.gif"
GIF_20260519 = DATA_DIR / "aemet_nacional_20260519_1039.gif"
RECOVERED_DARK_GREEN = (67, 131, 35)


def _disc_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    cy, cx = height // 2, width // 2
    radius = min(cy, cx) * 0.90
    yy, xx = np.ogrid[:height, :width]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


class TestImageRadarRegression(unittest.TestCase):
    def test_palette_exact_colors_map_as_expected(self) -> None:
        for rgb, expected in AEMET_PALETTE:
            image = np.array([[rgb]], dtype=np.uint8)
            tmp_path = ROOT_DIR / "tests" / "_tmp_palette_test.png"
            Image.fromarray(image, mode="RGB").save(tmp_path)
            try:
                result = image_to_dbz(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)

            actual = result[0, 0]
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual), msg=f"{rgb} should map to NaN")
            else:
                self.assertEqual(float(actual), float(expected), msg=f"{rgb} mapped incorrectly")

    def test_dark_green_echo_pixels_are_recovered(self) -> None:
        self.assertTrue(GIF_20260515.exists())
        with Image.open(GIF_20260515) as img:
            raw_rgb = np.array(img.convert("RGB"))
        disc = _disc_mask(raw_rgb.shape[:2])
        dark_green_mask = np.all(raw_rgb == RECOVERED_DARK_GREEN, axis=2)

        dbz = image_to_dbz(str(GIF_20260515))

        self.assertEqual(int((dark_green_mask & disc).sum()), 230)
        self.assertEqual(int(np.sum((dbz == 30.0) & dark_green_mask & disc)), 230)
        self.assertEqual(int(np.sum(np.isfinite(dbz) & dark_green_mask & ~disc)), 0)

    def test_background_masking_keeps_outside_disc_empty(self) -> None:
        dbz = image_to_dbz(str(GIF_20260515))
        disc = _disc_mask(dbz.shape)

        self.assertTrue(np.isnan(dbz[0, 0]))
        self.assertTrue(np.isnan(dbz[-1, -1]))
        self.assertEqual(int(np.sum(np.isfinite(dbz[~disc]))), 0)

    def test_class_counts_match_known_gifs(self) -> None:
        expected = {
            GIF_20260515: {12.0: 5220, 18.0: 3679, 24.0: 1160, 30.0: 274, 36.0: 6},
            GIF_20260519: {12.0: 485, 18.0: 193, 24.0: 44, 30.0: 10},
        }

        for gif_path, class_counts in expected.items():
            with self.subTest(gif=gif_path.name):
                self.assertTrue(gif_path.exists())
                dbz = image_to_dbz(str(gif_path))
                values, counts = np.unique(dbz[np.isfinite(dbz)], return_counts=True)
                actual = dict(zip(values.tolist(), counts.tolist()))
                self.assertEqual(actual, class_counts)


if __name__ == "__main__":
    unittest.main()
