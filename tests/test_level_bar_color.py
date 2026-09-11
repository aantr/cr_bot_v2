"""Level-bar team voting uses pixel counts, not weighted channel averages."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import predict_video_kalman as video


class LevelBarColorTests(unittest.TestCase):
    def setUp(self):
        for name in ("FIELD_COLOR_BLUE_MIN_DOMINANCE", "FIELD_COLOR_RED_MIN_DOMINANCE"):
            patcher = patch.object(video, name, 10)
            patcher.start()
            self.addCleanup(patcher.stop)

    def count(self, pixels):
        return video.count_level_bar_pixels(np.array([pixels], dtype=np.uint8))

    def test_tinted_pixels_threshold_inclusive_and_both_other_channels(self):
        counts = self.count([(110, 100, 99), (109, 100, 99), (110, 105, 0),
                             (99, 100, 110), (99, 100, 109), (0, 105, 110)])
        self.assertEqual(counts, (1, 1))

    def test_green_gray_white_black_and_channel_ties_do_not_vote(self):
        self.assertEqual(self.count([(0, 255, 0), (120, 120, 120), (255, 255, 255),
                                      (0, 0, 0), (200, 50, 200), (255, 255, 0),
                                      (0, 255, 255)]), (0, 0))

    def test_zero_threshold_still_requires_strict_channel_dominance(self):
        with patch.object(video, "FIELD_COLOR_BLUE_MIN_DOMINANCE", 0), \
             patch.object(video, "FIELD_COLOR_RED_MIN_DOMINANCE", 0):
            self.assertEqual(self.count([(101, 100, 100), (100, 100, 101),
                                          (100, 100, 100), (120, 0, 120)]), (1, 1))

    def test_red_and_blue_thresholds_can_be_tuned_independently(self):
        with patch.object(video, "FIELD_COLOR_BLUE_MIN_DOMINANCE", 20):
            self.assertEqual(self.count([(115, 100, 100), (100, 100, 115)]), (0, 1))
        with patch.object(video, "FIELD_COLOR_RED_MIN_DOMINANCE", 20):
            self.assertEqual(self.count([(115, 100, 100), (100, 100, 115)]), (1, 0))

    def test_more_weak_blue_pixels_win_over_one_bright_red_pixel(self):
        pixels = [(25, 10, 10)] * 5 + [(0, 0, 255)]
        counts = self.count(pixels)
        self.assertEqual(counts, (5, 1))
        self.assertEqual(video.level_bar_side(counts), "ally")
        # Mean red is higher, but means must no longer determine the team.
        mean = np.mean(pixels, axis=0)
        self.assertGreater(mean[2], mean[0])
        self.assertEqual(video.level_bar_side(self.count([(r, g, b) for b, g, r in pixels])), "enemy")

    def test_signed_subtraction_prevents_uint8_wraparound(self):
        self.assertEqual(self.count([(0, 0, 255), (255, 0, 0), (0, 255, 0)]), (1, 1))

    def test_missing_empty_and_tie_are_unknown_gray(self):
        for crop in (None, np.empty((0, 5, 3), dtype=np.uint8)):
            self.assertEqual(video.count_level_bar_pixels(crop), (0, 0))
        for counts in (None, (0, 0), (20, 20)):
            self.assertEqual(video.level_bar_side(counts), "unknown")
            self.assertEqual(video.track_color(3, counts), (190, 190, 190))

    def test_invalid_image_format_is_rejected(self):
        for crop in (np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 3, 3), dtype=float)):
            with self.assertRaises(ValueError):
                video.count_level_bar_pixels(crop)

    def test_field_fill_uses_the_same_votes_even_on_hash_cells(self):
        background = video.build_field_background()
        row = next(i for i, cells in enumerate(video.FIELD) if "#" in cells)
        col = video.FIELD[row].index("#")
        for counts in ((4, 1), (1, 4), (2, 2)):
            painted = video.draw_objects_on_field(
                background, [("blue_rect", 7, col + .5, row + .5)], (32, 18, 3), {7: counts})
            center = painted[row * video.FIELD_CELL_SIZE + video.FIELD_CELL_SIZE // 2,
                             col * video.FIELD_CELL_SIZE + video.FIELD_CELL_SIZE // 2]
            self.assertEqual(tuple(center), video.track_color(7, counts))

    def test_memory_retains_pixel_counts_for_kalman_only_tracks(self):
        memory = video.UnitTrackMemory(.9, (20., 10., 100.), "knight .9", (5, 1))
        self.assertEqual(video.level_bar_side(memory.level_bar_pixel_counts), "ally")
        self.assertNotEqual(video.track_color(1, (5, 1)), video.track_color(2, (5, 1)))


if __name__ == "__main__":
    unittest.main()
