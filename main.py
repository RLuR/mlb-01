import pandas as pd
import models.knn
import utils
import numpy as np

from sklearn.datasets import load_boston

from utils import train_and_evaluate_knn


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
        my_acc, my_std_dev = train_and_evaluate_knn(X, Y, k, splits, models.knn.classify, utils.get_accuracy)
        my_results.append([k, my_acc, my_std_dev])
        sklearn_acc, sklearn_std_dev = train_and_evaluate_knn(X, Y, k, splits, utils.train_and_classify_sklearn, utils.get_accuracy)
        sklearn_results.append([k, sklearn_acc, sklearn_std_dev])

    my_results = pd.DataFrame(data=my_results, columns=["k", "accuracy", "std_dev"],)
    sklearn_results = pd.DataFrame(data=sklearn_results, columns=["k", "accuracy", "std_dev"])


    print(f"My results: \n{my_results}")
    print(f"SKlearn results: \n{sklearn_results}")
    print(f"My highest accuracy: \n{my_results.loc[my_results['accuracy'].idxmax()]}")
    print(f"Sklearn highest accuracy: \n{sklearn_results.loc[sklearn_results['accuracy'].idxmax()]}")

    my_results.to_csv("results/my_classification_results.csv")
    sklearn_results.to_csv("results/sklearn_classification_results.csv")


def test_regressor():
    data = load_boston()
    X = pd.DataFrame(data.data)
    Y = pd.DataFrame(data.target)

    X = utils.normalize_features(X)

    my_results = []
    sklearn_results = []

    splits = 10

    # hyper parameter training
    for k in [1, 2, 3, 5, 9]:
        my_acc, my_std_dev = train_and_evaluate_knn(X, Y, k, splits, models.knn.regress, utils.get_mean_square_error)
        my_results.append([k, my_acc, my_std_dev])
        sklearn_err, sklearn_std_dev = train_and_evaluate_knn(X, Y, k, splits, utils.train_and_regress_sklearn, utils.get_mean_square_error)
        sklearn_results.append([k, sklearn_err, sklearn_std_dev])

    my_results = pd.DataFrame(data=my_results, columns=["k", "mse", "std_dev"],)
    sklearn_results = pd.DataFrame(data=sklearn_results, columns=["k", "mse", "std_dev"])

    print(f"My results: \n{my_results}")
    print(f"SKlearn results: \n{sklearn_results}")
    print(f"My best mse: \n{my_results.loc[my_results['mse'].idxmin()]}")
    print(f"Sklearn best mse: \n{sklearn_results.loc[sklearn_results['mse'].idxmin()]}")

    # save results

    my_results.to_csv("results/my_regression_results.csv")
    sklearn_results.to_csv("results/sklearn_regression_results.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_classifier()
    test_regressor()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
