from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=2)


def _load_x_and_y(path: Path):
    df = pd.read_csv(path, delimiter="\t", index_col="ID")
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return X, y


def main():
    base_data_path = Path("knn") / "data"
    data_folders = {
        "original_2f": base_data_path / "original" / "2_features",
        "normalized_2f": base_data_path / "normalized" / "2_features",
        "original_11f": base_data_path / "original" / "11_features",
        "normalized_11f": base_data_path / "normalized" / "11_features",
    }

    for data_name, data_folder in data_folders.items():
        classifiers = _train_classifiers(data_folder)
        _print_accuracy(classifiers, data_folder, data_name)
        _print_nearest_neighbors_and_distances(classifiers, data_folder, data_name)
        print("*"*150)


def _train_classifiers(data_folder: Path) -> dict[int, KNeighborsClassifier]:
    ks = [1, 3, 5, 7]
    X_training, y_training = _load_x_and_y(data_folder / "training.csv")
    return {k:KNeighborsClassifier(n_neighbors=k).fit(X_training, y_training) for k in ks}


def _print_accuracy(classifiers: dict[int, KNeighborsClassifier], data_folder: Path, data_name: str):
    X_testing, y_testing = _load_x_and_y(data_folder / "testing.csv")
    for k, classifier in classifiers.items():
        predictions = classifier.predict(X_testing)
        accuracy = accuracy_score(y_testing, predictions)
        print(f"[{data_name:14} {k=}] Measured accuracy: {accuracy:.2%}")


def _print_nearest_neighbors_and_distances(classifiers: dict[int, KNeighborsClassifier], data_folder: Path, data_name: str):
    X_training, _ = _load_x_and_y(data_folder / "training.csv")
    X_testing, _ = _load_x_and_y(data_folder / "testing.csv")

    for k, classifier in classifiers.items():
        distances, nearest_neighbors = classifier.kneighbors(X_testing, n_neighbors=k)
        predictions = classifier.predict(X_testing)
        for i, (neighbors, distances, prediction) in enumerate(zip(nearest_neighbors, distances, predictions)):
            neighbors = X_training.iloc[neighbors]
            print(
                f"[{data_name:14} {k=}] ",
                f"{X_testing.index[i]}: "
                f"predicted class={prediction} "
                f"nearest_neighbors={neighbors.index.values} "
                f"distances={distances}"
            )


if __name__ == "__main__":
    main()
