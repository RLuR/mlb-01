import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#Fixed rng seed for reproducability
import models.decisiontree

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

def get_r_value(true_results, predicted_results) -> float:
    rss = get_mean_square_error(true_results, predicted_results)
    tss = get_mean_square_error(true_results, np.mean(true_results))

    return 1-(rss/tss)

def get_gini_impurity(Y: pd.Series) -> float:
    if len(Y) == 0:
        return 1

    class_result = Y.mode()[0]
    amount_correct = len(Y[Y == class_result])
    amount_wrong = len(Y[Y != class_result])
    total_amount = len(Y)

    return 1 - (amount_correct/total_amount) ** 2 - (amount_wrong/total_amount) ** 2

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

def train_and_evaluate_decision_tree(X, Y, max_depth, splits):
    results = []
    for i in range(splits):
        X_train, Y_train, X_test, Y_test = monte_carlo_split(X, Y)

        tree = models.decisiontree.DecisionTree(max_depth=max_depth)
        tree.train(X_train, Y_train)
        Y_predict = X_test.apply(lambda test_row: tree.predict(test_row), axis=1).to_numpy()
        Y_test = Y_test.to_numpy()
        results.append(get_accuracy(Y_predict, Y_test))

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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
