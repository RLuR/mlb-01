from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import get_mean_square_error, get_r_value
from sklearn import metrics
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import linregress

def main():
    data = load_boston()
    X = pd.DataFrame(data.data)
    Y = pd.DataFrame(data.target)

    stats_x = X[5].to_numpy()


    regressor = LinearRegression()
    regressor.fit(stats_x.reshape(-1,1), Y)

    Y_predict = regressor.predict(stats_x.reshape(-1,1))

    my_mse = get_mean_square_error(Y_predict, Y)
    my_R2 = get_r_value(Y_predict, Y)

    sklearn_mse = metrics.mean_squared_error(Y_predict, Y)
    sklearn_R2 = metrics.r2_score(Y_predict, Y)

    print(f"My mse: {my_mse}")
    print(f"My R2: {my_R2}")
    print(f"sklearn mse: {sklearn_mse}")
    print(f"sklearn R2: {sklearn_R2}")

    all_mse = np.square(np.subtract(Y,Y_predict))

    #get p value
    stats_y = np.apply_along_axis(lambda x: x[0],1, Y.to_numpy())

    res = linregress(x=stats_x, y=stats_y)
    print(f"P-value: {res.pvalue}")

    #TODO Plot
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))

    ax[0].set_xlim([0, 200])
    ax[1].set_xlim([4, 9])
    sns.histplot(data=all_mse, ax= ax[0])

    plottdata = pd.DataFrame(columns=["crime_rate", "value"])
    plottdata["avg rooms per dwelling"] = X[5]
    plottdata["value"] = Y
    sns.regplot(x="avg rooms per dwelling", y="value", data=plottdata.sample(random_state=1, n=100), ax=ax[1])

    plt.show()

if __name__ == "__main__":
    main()