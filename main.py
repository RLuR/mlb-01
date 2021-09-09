import models.knn
import utils
import numpy as np


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

    X_train, Y_train, X_test, Y_test = utils.monte_carlo_split(X, Y)
    # train and test model
    Y_predict = models.knn.classify(X_train, Y_train, X_test, 5)

    print(Y_predict)
    print(Y_test)
    # evaluate


def test_regressor():
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_classifier()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
