import numpy as np
import pandas as pd


def classify(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, k: int) -> np.array:
    return np.apply_along_axis(lambda test_row: classify_row(x_train, y_train, test_row, k), axis=1, arr=x_test)


def classify_row(x_train: pd.DataFrame, y_train: pd.Series, test_row: pd.Series, k: int) -> int:
    closest_elements_idx = get_k_closest_training_data(x_train, test_row, k)
    return get_most_common_value(y_train.iloc[closest_elements_idx])


def regress(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, k: int) -> np.array:
    return np.apply_along_axis(lambda test_row: regress_row(x_train, y_train, test_row, k), axis=1, arr=x_test)


def regress_row(x_train: pd.DataFrame, y_train: pd.Series, test_row: pd.Series, k: int):
    closest_elements = get_k_closest_training_data(x_train, test_row, k)
    return np.mean(y_train.iloc[closest_elements])


def get_k_closest_training_data(x_train: pd.DataFrame, x_test: pd.Series, k) -> list:
    distances = np.apply_along_axis(lambda train_row: distance_function(train_row, x_test), axis=1, arr=x_train)
    # get the indices of the k lowest values
    return np.argsort(distances)[:k]


def distance_function(vector_1: pd.Series, vector_2: pd.Series) -> float:
    # Some numpy magic to get the euclidean distance
    # euclidean distance is defined as the L2 norm of the vectors
    return np.linalg.norm(vector_1 - vector_2, ord=2)


def get_most_common_value(y_train):
    # Uniform weight
    return y_train.mode()[0]
