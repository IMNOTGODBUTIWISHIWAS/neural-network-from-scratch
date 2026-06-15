import unittest

import numpy as np

from nn_scratch.math_utils import (
    accuracy_from_probabilities,
    deriv_sigmoid,
    mse_loss,
    sigmoid,
)


class TestMathUtils(unittest.TestCase):
    def test_sigmoid_zero_is_half(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=7)

    def test_sigmoid_large_positive_is_close_to_one(self):
        self.assertGreater(sigmoid(10.0), 0.99)

    def test_sigmoid_large_negative_is_close_to_zero(self):
        self.assertLess(sigmoid(-10.0), 0.01)

    def test_sigmoid_derivative_at_zero_is_quarter(self):
        self.assertAlmostEqual(deriv_sigmoid(0.0), 0.25, places=7)

    def test_mse_loss(self):
        y_true = np.array([1, 0, 0, 1], dtype=float)
        y_pred = np.array([0, 0, 0, 0], dtype=float)

        self.assertAlmostEqual(mse_loss(y_true, y_pred), 0.5, places=7)

    def test_accuracy_from_probabilities(self):
        y_true = np.array([1, 0, 1, 0], dtype=int)
        y_prob = np.array([0.9, 0.1, 0.7, 0.2], dtype=float)

        self.assertAlmostEqual(
            accuracy_from_probabilities(y_true, y_prob),
            1.0,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()