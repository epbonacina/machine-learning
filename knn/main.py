import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=2)


def _load_x_and_y(path: str):
    df = pd.read_csv(path, delimiter="\t", index_col="ID")
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return X, y


if __name__ == "__main__":
    BASE_DATA_PATH = os.path.join("knn", "data")
    DATA_FOLDERS = {
        "original_2f": os.path.join(BASE_DATA_PATH, "original", "2_features"),
        "original_11f": os.path.join(BASE_DATA_PATH, "original", "11_features"),
        "normalized_2f": os.path.join(BASE_DATA_PATH, "normalized", "2_features"),
        "normalized_11f": os.path.join(BASE_DATA_PATH, "normalized", "11_features"),
    }

    for data_name, data_folder in DATA_FOLDERS.items():
        training_data_path = os.path.join(data_folder, "training.csv")
        X_training, y_training = _load_x_and_y(training_data_path)

        testing_data_path = os.path.join(data_folder, "testing.csv")
        X_testing, y_testing = _load_x_and_y(testing_data_path)
        x_testing_indexes = X_testing.index

        for n in [1, 3, 5, 7]:
            classifier = KNeighborsClassifier(n_neighbors=n)
            classifier.fit(X_training, y_training)

            predictions = classifier.predict(X_testing)
            accuracy = accuracy_score(y_testing, predictions)

            distances, nearest_neighbors = classifier.kneighbors(X_testing, n_neighbors=n)

            print(f"[{data_name:14} {n=}] Measured accuracy: {accuracy:.2%}")

            for i, (neighbors, distances, prediction) in enumerate(zip(nearest_neighbors, distances, predictions)):
                neighbors = X_training.iloc[neighbors]
                print(
                    f"-> {x_testing_indexes[i]}: "
                    f"predicted class={prediction} "
                    f"nearest_neighbors={neighbors.index.values} "
                    f"distances={distances}"
                )
