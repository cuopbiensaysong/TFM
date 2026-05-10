import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.visualize import plot_trajectory_dimensions


class PlotTrajectoryDimensionsTest(unittest.TestCase):
    def test_creates_labeled_subplot_per_dimension(self):
        t_span = np.array([0.0, 1.0, 2.0])
        pred = np.array(
            [
                [0.0, 1.0],
                [0.5, 1.5],
                [1.0, 2.0],
            ]
        )
        groundtruth = np.array(
            [
                [0.1, 0.9],
                [0.4, 1.6],
                [0.9, 2.1],
            ]
        )

        fig = plot_trajectory_dimensions(pred, groundtruth, t_span=t_span, title="patient 0")

        self.assertEqual(len(fig.axes), 2)
        for dim, ax in enumerate(fig.axes):
            labels = [line.get_label() for line in ax.get_lines()]
            self.assertEqual(labels, ["Prediction", "Ground Truth"])
            self.assertEqual(ax.get_title(), f"Dimension {dim}")
            self.assertEqual(ax.get_xlabel(), "Time")
            self.assertEqual(ax.get_ylabel(), f"Dim {dim}")
            self.assertIsNotNone(ax.get_legend())


if __name__ == "__main__":
    unittest.main()
