from unittest import TestCase
from sklearn import metrics
import numpy as np

from utils import get_mean_square_error

class UtilsTest(TestCase):

    def test_custom_mse(self):
        truth = np.array([1,2,-3, 8, 100])
        prediction = np.array([1, 4, 2, -1, 50])

        my_mse = get_mean_square_error(truth,prediction)
        sklearn_mse = metrics.mean_squared_error(truth, prediction)

        assert abs(my_mse - sklearn_mse) <= 0.1

