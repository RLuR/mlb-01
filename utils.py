import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#Fixed rng seed for reproducability
rng = np.random.default_rng(1000)


def load_csv_data(path: str, seperator=";") -> pd.DataFrame:
    return pd.read_csv(path, sep=seperator)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def get_features_and_targets(df: pd.DataFrame, target_name="class") -> (pd.DataFrame, pd.Series):
    return df.drop(target_name, axis=1), df[target_name]


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df-df.min())/(df.max()-df.min())


def monte_carlo_split(X: pd.DataFrame, Y: pd.DataFrame, test_ratio=0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    test_size =  round(len(X) * test_ratio)
    random_indexes = rng.choice(len(X) - 1, size= test_size, replace=False)
    return X.drop(random_indexes),  Y.drop(random_indexes), X.iloc[random_indexes], Y.iloc[random_indexes]

def get_accuracy(true_results, predicted_results) -> float:
    matching_results = 0
    for index, value in enumerate(true_results):
        if value == predicted_results[index]:
            matching_results += 1

    return matching_results/len(true_results)

def get_mean_square_error(true_results, predicted_results) -> float:
    return np.square(np.subtract(true_results,predicted_results)).mean()


def train_and_evaluate_knn(X, Y, k, splits, classifier_method, evaluation_function):
    results = []
    for i in range(splits):
        X_train, Y_train, X_test, Y_test = monte_carlo_split(X, Y)
        # train and test model
        Y_predict = classifier_method(X_train, Y_train, X_test, k)

        Y_test = Y_test.to_numpy()
        # evaluate
        results.append(evaluation_function(Y_predict, Y_test))
    results = pd.Series(results)
    return results.mean(), results.std()


def train_and_classify_sklearn(X_train, Y_train, X_test, k):
    knn_model = KNeighborsClassifier(n_neighbors=k,
                                     metric="euclidean",
                                     weights="uniform")
    knn_model = knn_model.fit(X_train, Y_train)
    return knn_model.predict(X_test)

def train_and_regress_sklearn(X_train, Y_train, X_test, k):
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model = knn_model.fit(X_train, Y_train)
    return knn_model.predict(X_test)