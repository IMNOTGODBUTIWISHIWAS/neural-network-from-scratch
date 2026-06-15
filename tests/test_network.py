import unittest

import numpy as np

from nn_scratch.datasets import load_article_training_data
from nn_scratch.network import TinyNeuralNetwork


class TestTinyNeuralNetwork(unittest.TestCase):
    def test_forward_pass_matches_article_style_example(self):
        """
        Recreate the worked example where all weights are 1 and all biases are 0.

        Input: Alice = [-2, -1]
        Expected article-style values:
        h1 ≈ 0.0474
        h2 ≈ 0.0474
        y_pred ≈ 0.524
        """

        network = TinyNeuralNetwork(seed=0)
        network.set_parameters(
            w1=1,
            w2=1,
            w3=1,
            w4=1,
            w5=1,
            w6=1,
            b1=0,
            b2=0,
            b3=0,
        )
        details = network.forward_detailed(np.array([-2.0, -1.0]))
        self.assertAlmostEqual(details["h1"], 0.0474, places=3)
        self.assertAlmostEqual(details["h2"], 0.0474, places=3)
        self.assertAlmostEqual(details["y_pred"], 0.524, places=3)

    def test_gradient_for_w1_matches_article_style_example(self):
        """
        In the article's hand worked exampple dl/dw1 ≈ 0.0214.
        """
        network = TinyNeuralNetwork(seed=0)
        network.set_parameters(
            w1=1,
            w2=1,
            w3=1,
            w4=1,
            w5=1,
            w6=1,
            b1=0,
            b2=0,
            b3=0,
        )
        gradients, _ = network.compute_gradients(np.array([-2.0, -1.0]), y_true=1.0)
        self.assertAlmostEqual(gradients["w1"], 0.0214, places=3)

    def test_training_reduces_loss(self):
        X, y, _ = load_article_training_data()
        network = TinyNeuralNetwork(seed=42)
        initial_loss = network.loss_on_dataset(X, y)
        network.train(
            X,
            y,
            epochs=1000,
            learning_rate=0.1,
            report_every=1000,
            shuffle=False,
            verbose=False,
        )
        final_loss = network.loss_on_dataset(X, y)
        self.assertLess(final_loss, initial_loss)
    
    def test_predict_probabilities_shape(self):
        X, _, _ = load_article_training_data()
        network = TinyNeuralNetwork(seed=42)
        predictions = network.predict_probabilities(X)
        self.assertEqual(predictions.shape, (len(X),))

    if __name__ == "__main__":
        unittest.main()