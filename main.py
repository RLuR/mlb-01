import pandas as pd

import models.knn
import utils
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def test_classifier():
    target_column = "class"

    # load dataset
    df = utils.load_csv_data("data/iris.csv")
    # clean dataset
    df = utils.clean_dataset(df)

    labels = np.unique(df["class"])
    df["class"].replace({labels[0]: 0, labels[1]: 1, labels[2]: 2}, inplace=True)

    X, Y = utils.get_features_and_targets(df)

    # normalize features
    X = utils.normalize_features(X)

    my_accuracies = pd.Series()
    sk_learn_accuracies = pd.Series()
    splits = 10

    for k in range(1,15):
        my_accuracies._set_value(k, train_and_evaluate_knn(X, Y, k, splits, models.knn.classify))
        sk_learn_accuracies._set_value(k, train_and_evaluate_knn(X, Y, k, splits, train_and_predict_sklearn))

    print(f"My accuracies: \n{my_accuracies}")
    print(f"SKlearn accuracies: \n{sk_learn_accuracies}")
    print(f"My best accuracy: {my_accuracies.max()}, with k = {my_accuracies.idxmax() }")
    print(f"sklearn accuracy: {sk_learn_accuracies.max()}, with k = {sk_learn_accuracies.idxmax()}")


def train_and_evaluate_knn(X, Y, k, splits, classifier_method):
    accuracies = []
    for i in range(splits):
        X_train, Y_train, X_test, Y_test = utils.monte_carlo_split(X, Y)
        # train and test model
        Y_predict = classifier_method(X_train, Y_train, X_test, k)

        Y_test = Y_test.to_numpy()
        # evaluate
        accuracies.append(utils.get_accuracy(Y_predict, Y_test))
    accuracies = pd.Series(accuracies)
    overall_acc = accuracies.mean()
    return overall_acc


def train_and_predict_sklearn(X_train, Y_train, X_test, k):
    knn_model = KNeighborsClassifier(n_neighbors=k,
                                     # hyperparameter k, usually odd, thehigher the smoother the decision surface
                                     metric="euclidean",  # metric used for distance calculation (more later)
                                     weights="uniform")  # if estimation should be weighted by distance (more later)
    knn_model = knn_model.fit(X_train, Y_train)
    return knn_model.predict(X_test)


def test_regressor():
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_classifier()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
