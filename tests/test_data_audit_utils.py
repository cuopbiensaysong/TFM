import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils.data_audit import compute_pair_variables, summarize_array  # pylint: disable=import-error


class TestDataAuditUtils(unittest.TestCase):
    def test_compute_pair_variables_and_stats(self):
        x0 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
        x1 = np.array([[2.0, 4.0], [7.0, 9.0]], dtype=np.float32)
        t0 = np.array([1.0, 2.0], dtype=np.float32)
        t1 = np.array([3.0, 4.0], dtype=np.float32)

        delta_time, delta_x, velocity, zero_dt = compute_pair_variables(x0, x1, t0, t1)

        np.testing.assert_allclose(delta_time, np.array([2.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(
            delta_x,
            np.array([[1.0, 2.0], [4.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            velocity,
            np.array([[0.5, 1.0], [2.0, 2.0]], dtype=np.float32),
        )
        self.assertEqual(zero_dt, 0)

        stats = summarize_array(delta_time)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["std"], 0.0)
        self.assertAlmostEqual(stats["min"], 2.0)
        self.assertAlmostEqual(stats["max"], 2.0)


if __name__ == "__main__":
    unittest.main()
