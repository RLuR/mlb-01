import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import utils

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df-df.min())/(df.max()-df.min())


def normalization():

    hippo_weight = np.random.randint(low=1000, high= 1900, size=100)
    hippo_shoulder_height = np.random.randint(low=110, high=170, size=100)

    hippo_dataset = pd.DataFrame(columns = ["weight", "shoulder_height"])
    hippo_dataset["weight"] = hippo_weight
    hippo_dataset["shoulder_height"] = hippo_shoulder_height

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax[0].set_ylim([0, 2000])
    ax[0].set_xlim([0, 2000])
    sns.scatterplot(data=hippo_dataset, x="shoulder_height", y="weight", ax=ax[0]).set_title("raw")

    normalized_features = normalize_features(hippo_dataset)

    sns.scatterplot(data=normalized_features, x="shoulder_height", y="weight", ax=ax[1]).set_title("normalized")


def one_hot_encoding():
    data = pd.read_csv("data/wine_cleaned.csv", delimiter=";")
    print(f"Seasons: {data['season'].unique()}")
    one_hot_encoded = pd.get_dummies(data["season"], prefix='season')
    print(one_hot_encoded)

if __name__ == "__main__":

    normalization()

    one_hot_encoding()

    plt.show()