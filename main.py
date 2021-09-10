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

    my_results = []
    sklearn_results = []

    splits = 10

    # hyper parameter training
    for k in range(1,15):
        my_acc, my_std_dev = train_and_evaluate_knn(X, Y, k, splits, models.knn.classify)
        my_results.append([k, my_acc, my_std_dev])
        sklearn_acc, sklearn_std_dev = train_and_evaluate_knn(X, Y, k, splits, models.knn.classify)
        sklearn_results.append([k, sklearn_acc, sklearn_std_dev])

    my_results = pd.DataFrame(data=my_results, columns=["k", "accuracy", "std_dev"],)
    sklearn_results = pd.DataFrame(data=sklearn_results, columns=["k", "accuracy", "std_dev"])


    print(f"My results: \n{my_results}")
    print(f"SKlearn results: \n{sklearn_results}")
    print(f"My highest accuracy: \n{my_results.loc[my_results['accuracy'].idxmax()]}")
    print(f"Sklearn highest accuracy: \n{sklearn_results.loc[sklearn_results['accuracy'].idxmax()]}")

def train_and_evaluate_knn(X, Y, k, splits, classifier_method):
    results = []
    for i in range(splits):
        X_train, Y_train, X_test, Y_test = utils.monte_carlo_split(X, Y)
        # train and test model
        Y_predict = classifier_method(X_train, Y_train, X_test, k)

        Y_test = Y_test.to_numpy()
        # evaluate
        results.append(utils.get_accuracy(Y_predict, Y_test))
    results = pd.Series(results)
    return results.mean(), results.std()


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
