import pandas as pd
import models.decisiontree
import utils
import numpy as np

from utils import train_and_evaluate_decision_tree


def test_tree():
    target_column = "class"

    # load dataset
    df = utils.load_csv_data("data/heart.csv", seperator="," )
    # clean dataset

    df = df[["sex", "fbs", "restecg", "exang", "target"]]
    df["restecg"] = df["restecg"].replace(2, 1)

    # normalize features
    X, Y = utils.get_features_and_targets(df, "target")

    my_results = []
    splits = 10

    # hyper parameter training
    for max_depth in range(0,5):
        my_acc, my_std_dev = train_and_evaluate_decision_tree(X, Y, max_depth, splits)
        my_results.append([max_depth, my_acc, my_std_dev])

    my_results = pd.DataFrame(data=my_results, columns=["k", "accuracy", "std_dev"],)


    print(f"My results: \n{my_results}")
    print(f"My highest accuracy: \n{my_results.loc[my_results['accuracy'].idxmax()]}")

    my_results.to_csv("results/my_decision_tree_results.csv")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_tree()