import argparse
import csv
import json
from pathlib import Path

from .datasets import (
    load_article_prediction_examples,
    load_article_training_data,
    load_easy_points_data,
)
from .network import TinyNeuralNetwork
from .visualize import generate_all_figures


def save_history_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "accuracy"])

        for epoch, loss, accuracy in zip(
            history["epoch"],
            history["loss"],
            history["accuracy"],
        ):
            writer.writerow([epoch, loss, accuracy])


def save_predictions_csv(names, probabilities, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "probability_class_1", "predicted_label"])

        for name, probability in zip(names, probabilities):
            predicted_label = int(probability >= 0.5)
            writer.writerow([name, f"{probability:.6f}", predicted_label])


def save_parameters_json(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(model.parameters(), file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny neural network from scratch."
    )

    parser.add_argument(
        "--dataset",
        choices=["people", "easy_points"],
        default="people",
        help="Which dataset to train on.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="How many times to loop over the full dataset.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="SGD learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for parameter initialization.",
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="How often to print loss/accuracy.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples each epoch. Leave off to stay closer to the article.",
    )

    return parser.parse_args()


def load_dataset(which):
    if which == "people":
        return load_article_training_data()

    if which == "easy_points":
        return load_easy_points_data()

    raise ValueError(f"Unknown dataset: {which}")


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts" / args.dataset
    figures_dir = project_root / "figures" / "generated" / args.dataset

    X_train, y_train, names = load_dataset(args.dataset)

    print(f"Training dataset: {args.dataset}")
    print(f"Samples: {len(X_train)}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print(f"Shuffle each epoch: {args.shuffle}")
    print()

    model = TinyNeuralNetwork(seed=args.seed)

    history = model.train(
        X=X_train,
        y=y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        report_every=args.report_every,
        shuffle=args.shuffle,
        verbose=True,
    )

    save_history_csv(history, artifacts_dir / "history.csv")
    save_parameters_json(model, artifacts_dir / "parameters.json")

    probabilities_on_train = model.predict_probabilities(X_train)

    save_predictions_csv(
        names,
        probabilities_on_train,
        artifacts_dir / "train_predictions.csv",
    )

    figure_paths = generate_all_figures(
        model=model,
        history=history,
        X=X_train,
        y=y_train,
        names=names,
        output_dir=figures_dir,
        prefix=args.dataset,
    )

    print()
    print("Saved artifacts:")
    print(f" - {artifacts_dir / 'history.csv'}")
    print(f" - {artifacts_dir / 'parameters.json'}")
    print(f" - {artifacts_dir / 'train_predictions.csv'}")

    print()
    print("Saved figures:")
    for key, path in figure_paths.items():
        print(f" - {key}: {path}")

    if args.dataset == "people":
        print()
        print("Prediction examples:")

        X_predict, predict_names = load_article_prediction_examples()
        predict_probabilities = model.predict_probabilities(X_predict)

        save_predictions_csv(
            predict_names,
            predict_probabilities,
            artifacts_dir / "article_prediction_examples.csv",
        )

        for name, probability in zip(predict_names, predict_probabilities):
            predicted_label = "class 1" if probability >= 0.5 else "class 0"
            print(
                f" - {name}: probability = {probability:.3f} -> {predicted_label}"
            )

        print()
        print(
            f"Saved extra file: "
            f"{artifacts_dir / 'article_prediction_examples.csv'}"
        )


if __name__ == "__main__":
    main()