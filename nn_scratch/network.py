import numpy as np

from .math_utils import (
    accuracy_from_probabilities,
    deriv_sigmoid,
    mse_loss,
    sigmoid,
)


class TinyNeuralNetwork:
    """
    A tiny educational neural network with this shape:

        2 inputs -> 2 hidden neurons -> 1 output neuron

    Parameter meaning:

        w1, w2: input -> h1
        w3, w4: input -> h2
        w5, w6: hidden -> output

        b1, b2, b3: biases

    The naming is intentionally explicit so the code maps clearly
    to the math from Victor Zhou's article.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Start with random weights and biases.

        The network does not know anything yet. These random numbers
        are the starting point before training.
        """
        self.w1 = float(self.rng.normal())
        self.w2 = float(self.rng.normal())
        self.w3 = float(self.rng.normal())
        self.w4 = float(self.rng.normal())
        self.w5 = float(self.rng.normal())
        self.w6 = float(self.rng.normal())

        self.b1 = float(self.rng.normal())
        self.b2 = float(self.rng.normal())
        self.b3 = float(self.rng.normal())

    def set_parameters(
        self,
        w1=None,
        w2=None,
        w3=None,
        w4=None,
        w5=None,
        w6=None,
        b1=None,
        b2=None,
        b3=None,
    ):
        """
        Convenience helper for tests and hand-worked math experiments.

        This lets us manually set weights and biases to known values.
        """
        if w1 is not None:
            self.w1 = float(w1)
        if w2 is not None:
            self.w2 = float(w2)
        if w3 is not None:
            self.w3 = float(w3)
        if w4 is not None:
            self.w4 = float(w4)
        if w5 is not None:
            self.w5 = float(w5)
        if w6 is not None:
            self.w6 = float(w6)

        if b1 is not None:
            self.b1 = float(b1)
        if b2 is not None:
            self.b2 = float(b2)
        if b3 is not None:
            self.b3 = float(b3)

    def parameters(self):
        """
        Return all trainable parameters as a plain dictionary.
        """
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "w4": self.w4,
            "w5": self.w5,
            "w6": self.w6,
            "b1": self.b1,
            "b2": self.b2,
            "b3": self.b3,
        }

    def forward_detailed(self, x):
        """
        Run a forward pass and return all intermediate values.

        Feedforward means:

            input numbers -> hidden neurons -> output prediction

        This method returns the hidden values too, not only the final
        prediction, because seeing the intermediate values helps us learn.
        """
        x = np.asarray(x, dtype=float)

        if x.shape != (2,):
            raise ValueError("x must contain exactly 2 input values.")

        # Hidden neuron h1
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        # Hidden neuron h2
        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        # Output neuron o1
        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)

        return {
            "x": x,
            "sum_h1": float(sum_h1),
            "h1": float(h1),
            "sum_h2": float(sum_h2),
            "h2": float(h2),
            "sum_o1": float(sum_o1),
            "y_pred": float(o1),
        }

    def feedforward(self, x):
        """
        Return only the final prediction.
        """
        return self.forward_detailed(x)["y_pred"]

    def predict_probabilities(self, X):
        """
        Predict probabilities for one sample or many samples.

        If X is one row, return one prediction.
        If X is many rows, return one prediction per row.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return np.asarray([self.feedforward(X)], dtype=float)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n_samples, 2).")

        return np.asarray([self.feedforward(row) for row in X], dtype=float)

    def predict_classes(self, X, threshold=0.5):
        """
        Convert probabilities into 0/1 labels.

        Example:
            probability 0.80 -> class 1
            probability 0.20 -> class 0
        """
        probabilities = self.predict_probabilities(X)
        return (probabilities >= threshold).astype(int)

    def compute_gradients(self, x, y_true):
        """
        Compute gradients for one training example using backpropagation.

        A gradient answers:

            "If I change this weight or bias a tiny bit,
             how much does the loss change?"

        Returns:
            gradients: dictionary with dL/d(parameter)
            details: cached forward-pass values
        """
        y_true = float(y_true)

        details = self.forward_detailed(x)

        x = details["x"]
        sum_h1 = details["sum_h1"]
        h1 = details["h1"]
        sum_h2 = details["sum_h2"]
        h2 = details["h2"]
        sum_o1 = details["sum_o1"]
        y_pred = details["y_pred"]

        # Derivative of loss with respect to prediction.
        d_L_d_ypred = -2.0 * (y_true - y_pred)

        # Reuse sigmoid derivatives at each neuron.
        d_sigmoid_o1 = deriv_sigmoid(sum_o1)
        d_sigmoid_h1 = deriv_sigmoid(sum_h1)
        d_sigmoid_h2 = deriv_sigmoid(sum_h2)

        # Derivatives for output neuron o1.
        d_ypred_d_w5 = h1 * d_sigmoid_o1
        d_ypred_d_w6 = h2 * d_sigmoid_o1
        d_ypred_d_b3 = d_sigmoid_o1

        d_ypred_d_h1 = self.w5 * d_sigmoid_o1
        d_ypred_d_h2 = self.w6 * d_sigmoid_o1

        # Derivatives for hidden neuron h1.
        d_h1_d_w1 = x[0] * d_sigmoid_h1
        d_h1_d_w2 = x[1] * d_sigmoid_h1
        d_h1_d_b1 = d_sigmoid_h1

        # Derivatives for hidden neuron h2.
        d_h2_d_w3 = x[0] * d_sigmoid_h2
        d_h2_d_w4 = x[1] * d_sigmoid_h2
        d_h2_d_b2 = d_sigmoid_h2

        gradients = {
            "w1": float(d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1),
            "w2": float(d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2),
            "w3": float(d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3),
            "w4": float(d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4),
            "w5": float(d_L_d_ypred * d_ypred_d_w5),
            "w6": float(d_L_d_ypred * d_ypred_d_w6),
            "b1": float(d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1),
            "b2": float(d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2),
            "b3": float(d_L_d_ypred * d_ypred_d_b3),
        }

        return gradients, details

    def apply_gradients(self, gradients, learning_rate):
        """
        Update each parameter with stochastic gradient descent.

        SGD rule:

            parameter = parameter - learning_rate * gradient

        In plain English:
        move each weight/bias a small step in the direction that
        should reduce the loss.
        """
        self.w1 -= learning_rate * gradients["w1"]
        self.w2 -= learning_rate * gradients["w2"]
        self.w3 -= learning_rate * gradients["w3"]
        self.w4 -= learning_rate * gradients["w4"]
        self.w5 -= learning_rate * gradients["w5"]
        self.w6 -= learning_rate * gradients["w6"]

        self.b1 -= learning_rate * gradients["b1"]
        self.b2 -= learning_rate * gradients["b2"]
        self.b3 -= learning_rate * gradients["b3"]

    def loss_on_dataset(self, X, y):
        """
        Compute full-dataset loss.
        """
        y_pred = self.predict_probabilities(X)
        return float(mse_loss(y, y_pred))

    def accuracy_on_dataset(self, X, y, threshold=0.5):
        """
        Compute full-dataset classification accuracy.
        """
        y_pred = self.predict_probabilities(X)
        return accuracy_from_probabilities(y, y_pred, threshold=threshold)

    def train(
        self,
        X,
        y,
        epochs=1000,
        learning_rate=0.1,
        report_every=50,
        shuffle=False,
        verbose=True,
    ):
        """
        Train with stochastic gradient descent.

        If shuffle=False, the sample order stays fixed.
        That matches the article more closely and makes learning easier
        to trace by hand.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n_samples, 2).")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
        }

        indices = np.arange(len(X))

        for epoch in range(epochs):
            if shuffle:
                self.rng.shuffle(indices)

            for index in indices:
                gradients, _ = self.compute_gradients(X[index], y[index])
                self.apply_gradients(gradients, learning_rate)

            human_epoch = epoch + 1

            should_report = (
                human_epoch == 1
                or human_epoch % report_every == 0
                or human_epoch == epochs
            )

            if should_report:
                loss = self.loss_on_dataset(X, y)
                accuracy = self.accuracy_on_dataset(X, y)

                history["epoch"].append(human_epoch)
                history["loss"].append(loss)
                history["accuracy"].append(accuracy)

                if verbose:
                    print(
                        f"epoch {human_epoch:4d} | "
                        f"loss = {loss:.6f} | "
                        f"accuracy = {accuracy:.3f}"
                    )

        return history