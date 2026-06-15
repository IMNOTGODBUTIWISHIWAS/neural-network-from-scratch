import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_csv_dataset(
    path,
    feature_columns,
    label_column="label",
    name_column="name",
):
    """
    Load a small CSV dataset using Python's built-in csv module.

    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) or None
        names: list of row names
    """
    path = Path(path)

    features = []
    labels = []
    names = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            feature_row = [float(row[column]) for column in feature_columns]
            features.append(feature_row)

            if label_column is not None:
                labels.append(int(row[label_column]))

            if name_column is not None and name_column in row:
                names.append(row[name_column])

    X = np.asarray(features, dtype=float)
    y = None if label_column is None else np.asarray(labels, dtype=float)

    return X, y, names


def load_article_training_data():
    """
    Load the tiny training set inspired by the article, using shifted features.
    """
    return load_csv_dataset(
        DATA_DIR / "people_train.csv",
        feature_columns=["x1_weight_shifted", "x2_height_shifted"],
        label_column="label",
        name_column="name",
    )


def load_article_prediction_examples():
    """
    Load Emily and Frank from the article-style prediction examples.
    """
    X, _, names = load_csv_dataset(
        DATA_DIR / "people_predict.csv",
        feature_columns=["x1_weight_shifted", "x2_height_shifted"],
        label_column=None,
        name_column="name",
    )

    return X, names


def load_easy_points_data():
    """
    Load a simple 2D dataset that is useful for plots.
    """
    return load_csv_dataset(
        DATA_DIR / "easy_points.csv",
        feature_columns=["x1", "x2"],
        label_column="label",
        name_column="name",
    )