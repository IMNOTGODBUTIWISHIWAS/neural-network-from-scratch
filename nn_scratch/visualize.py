from pathlib import Path

import matplotlib

# Im gonna use a non interactive backend so saving plots works nicely on any machine and shi
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .math_utils import deriv_sigmoid, sigmoid


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_sigmoid(output_path):
    """
    Save a plot of the sigmoid function.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-10, 10, 400)
    y = sigmoid(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_title("Sigmoid activation")
    ax.set_xlabel("x")
    ax.set_ylabel("sigmoid(x)")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sigmoid_derivative(output_path):
    """
    Save a plot of the derivative of the sigmoid function.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-10, 10, 400)
    y = deriv_sigmoid(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_title("Derivative of sigmoid")
    ax.set_xlabel("x")
    ax.set_ylabel("sigmoid'(x)")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_loss_curve(history, output_path):
    """
    Save a line plot of loss versus epoch.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["loss"])
    ax.set_title("Training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_accuracy_curve(history, output_path):
    """
    Save a line plot of accuracy versus epoch.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["accuracy"])
    ax.set_title("Training accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_decision_boundary(model, X, y, names, output_path, title):
    """
    Save a 2D decision boundary plot.

    because the network has only 2 input features, we can draw a grid over
    the input space and color each point by the model's predicted probability
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    x1_min = X[:, 0].min() - 1.0
    x1_max = X[:, 0].max() + 1.0
    x2_min = X[:, 1].min() - 1.0
    x2_max = X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 250),
        np.linspace(x2_min, x2_max, 250),
    )

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    probabilities = model.predict_probabilities(grid_points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))

    contour = ax.contourf(
        xx,
        yy,
        probabilities,
        levels=20,
        cmap="coolwarm",
        alpha=0.65,
    )

    ax.contour(
        xx,
        yy,
        probabilities,
        levels=[0.5],
        colors="black",
        linewidths=2,
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="black",
        s=90,
    )

    if names:
        for name, x1, x2 in zip(names, X[:, 0], X[:, 1]):
            ax.annotate(
                name,
                (x1, x2),
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("predicted probability of class 1")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_all_figures(model, history, X, y, names, output_dir, prefix):
    """
    Create all standard figures for one training run
    """
    output_dir = ensure_directory(output_dir)

    paths = {
        "sigmoid": output_dir / f"{prefix}_sigmoid.png",
        "sigmoid_derivative": output_dir / f"{prefix}_sigmoid_derivative.png",
        "loss": output_dir / f"{prefix}_loss.png",
        "accuracy": output_dir / f"{prefix}_accuracy.png",
        "decision_boundary": output_dir / f"{prefix}_decision_boundary.png",
    }

    plot_sigmoid(paths["sigmoid"])
    plot_sigmoid_derivative(paths["sigmoid_derivative"])
    plot_loss_curve(history, paths["loss"])
    plot_accuracy_curve(history, paths["accuracy"])

    plot_decision_boundary(
        model=model,
        X=X,
        y=y,
        names=names,
        output_path=paths["decision_boundary"],
        title=f"Decision boundary: {prefix}",
    )

    return paths