import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def deriv_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def mse_loss(y_true, y_pred):
    """
    Mean squared error loss.

    y_true and y_pred should be arrays of the same length.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean((y_true - y_pred) ** 2)


def accuracy_from_probabilities(y_true, y_prob, threshold=0.5):
    """
    Turn probabilities into 0/1 labels and compute accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    return float(np.mean(y_pred == y_true))