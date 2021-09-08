import numpy as np
import pandas as pd


def predict(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, k: int) -> list:
    results = []
    for index, row in x_test.iterrows():
        closest_elements = get_k_closest_training_data(x_train, row, k)
        results.append(get_most_common_value(y_train.iloc[closest_elements]))
    return results


def get_k_closest_training_data(x_train: pd.DataFrame, x_test: pd.Series, k) -> list:
    distances = np.apply_along_axis(lambda train_row: distance_function(train_row, x_test), axis=1, arr=x_train)
    # get the indices of the k lowest values
    return np.argsort(distances)[:k]


def distance_function(vector_1, vector_2) -> float:
    # Some numpy magic to get the euclidean distance
    return np.linalg.norm(vector_1 - vector_2)

def get_most_common_value(y_train):
    return y_train.mode()[0]
