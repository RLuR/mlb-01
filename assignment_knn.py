import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Fixed rng seed for reproducability
rng = np.random.default_rng(1000)


def test_classifier():
    target_column = "class"

    # load dataset
    df = load_csv_data("data/iris.csv")
    # clean dataset
    df = clean_dataset(df)

    labels = np.unique(df["class"])
    df["class"].replace({labels[0]: 0, labels[1]: 1, labels[2]: 2}, inplace=True)

    X, Y = get_features_and_targets(df)

    # normalize features
    X = normalize_features(X)

    my_results = []
    sklearn_results = []

    splits = 10

    # hyper parameter training
    for k in range(1, 15):
        my_acc, my_std_dev = train_and_evaluate_knn(X, Y, k, splits, classify, get_accuracy)
        my_results.append([k, my_acc, my_std_dev])
        sklearn_acc, sklearn_std_dev = train_and_evaluate_knn(X, Y, k, splits, train_and_classify_sklearn, get_accuracy)
        sklearn_results.append([k, sklearn_acc, sklearn_std_dev])

    my_results = pd.DataFrame(data=my_results, columns=["k", "accuracy", "std_dev"], )
    sklearn_results = pd.DataFrame(data=sklearn_results, columns=["k", "accuracy", "std_dev"])

    print(f"My results: \n{my_results}")
    print(f"SKlearn results: \n{sklearn_results}")
    print(f"My highest accuracy: \n{my_results.loc[my_results['accuracy'].idxmax()]}")
    print(f"Sklearn highest accuracy: \n{sklearn_results.loc[sklearn_results['accuracy'].idxmax()]}")

    my_results.to_csv("results/my_classification_results.csv", index=False)
    sklearn_results.to_csv("results/sklearn_classification_results.csv", index=False)


def test_regressor():
    data = load_boston()
    X = pd.DataFrame(data.data)
    Y = pd.DataFrame(data.target)

    X = normalize_features(X)

    my_results = []
    sklearn_results = []

    splits = 10

    # hyper parameter training
    for k in [1, 2, 3, 5, 9]:
        my_acc, my_std_dev = train_and_evaluate_knn(X, Y, k, splits, regress, get_mean_square_error)
        my_results.append([k, my_acc, my_std_dev])
        sklearn_err, sklearn_std_dev = train_and_evaluate_knn(X, Y, k, splits, train_and_regress_sklearn,
                                                              get_mean_square_error)
        sklearn_results.append([k, sklearn_err, sklearn_std_dev])

    my_results = pd.DataFrame(data=my_results, columns=["k", "mse", "std_dev"], )
    sklearn_results = pd.DataFrame(data=sklearn_results, columns=["k", "mse", "std_dev"])

    print(f"My results: \n{my_results}")
    print(f"SKlearn results: \n{sklearn_results}")
    print(f"My best mse: \n{my_results.loc[my_results['mse'].idxmin()]}")
    print(f"Sklearn best mse: \n{sklearn_results.loc[sklearn_results['mse'].idxmin()]}")

    # save results

    my_results.to_csv("results/my_regression_results.csv", index=False)
    sklearn_results.to_csv("results/sklearn_regression_results.csv", index=False)


def load_csv_data(path: str, seperator=";") -> pd.DataFrame:
    return pd.read_csv(path, sep=seperator)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def get_features_and_targets(df: pd.DataFrame, target_name="class") -> (pd.DataFrame, pd.Series):
    return df.drop(target_name, axis=1), df[target_name]


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def monte_carlo_split(X: pd.DataFrame, Y: pd.DataFrame, test_ratio=0.2) -> (
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    test_size = round(len(X) * test_ratio)
    random_indexes = rng.choice(len(X) - 1, size=test_size, replace=False)
    return X.drop(random_indexes), Y.drop(random_indexes), X.iloc[random_indexes], Y.iloc[random_indexes]


def get_accuracy(true_results, predicted_results) -> float:
    matching_results = 0
    for index, value in enumerate(true_results):
        if value == predicted_results[index]:
            matching_results += 1

    return matching_results / len(true_results)


def get_mean_square_error(true_results, predicted_results) -> float:
    return np.square(np.subtract(true_results, predicted_results)).mean()


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_classifier()
    test_regressor()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
