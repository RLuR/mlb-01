import pandas as pd
from numpy.random import default_rng

rng = default_rng(1000)


def load_csv_data(path: str, seperator=";") -> pd.DataFrame:
    return pd.read_csv(path, sep=seperator)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def get_features_and_targets(df: pd.DataFrame, target_name="class") -> (pd.DataFrame, pd.Series):
    return df.drop(target_name, axis=1), df[target_name]


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df-df.min())/(df.max()-df.min())


def monte_carlo_split(X: pd.DataFrame, Y: pd.DataFrame, test_ratio = 0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    test_size =  round(len(X) * test_ratio)
    random_indexes = rng.choice(len(X) - 1, size= test_size, replace=False)
    return X.drop(random_indexes),  Y.drop(random_indexes), X.iloc[random_indexes], Y.iloc[random_indexes]

def get_accuracy(true_results, predicted_results) -> float:
    matching_results = 0
    for index, value in enumerate(true_results):
        if value == predicted_results[index]:
            matching_results += 1

    return matching_results/len(true_results)


