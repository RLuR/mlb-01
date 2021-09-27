from abc import ABC
import pandas as pd
import numpy as np
from utils import get_gini_impurity


class DecisionTree:

    def __init__(self, max_depth = 3):
        self.root = DecisionNode(0, max_depth=max_depth)
        pass

    def train(self, X: pd.DataFrame, Y: pd.Series):
        self.root.train(X,Y)

    def predict(self, X: pd.Series) -> int:
        return self.root.predict(X)

    def print(self):
        self.root.print()


class Node(ABC):
    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

class DecisionNode(Node):
    def __init__(self, depth, max_depth):
        self.depth = depth
        self.split_criterion = None
        # For now only binary data
        self.split_value = 0.5
        self.max_depth = max_depth

    def train(self, X, Y):
        self.get_split_criterion(X, Y)
        self.split(X,Y)
        pass

    def get_split_criterion(self, X, Y):
        best_gini = 1
        for columnname in X.head():
            X_left = X.loc[X[columnname] <= self.split_value]
            X_right = X.loc[X[columnname] > self.split_value]
            left_idx = X_left.index.values
            right_idx = X_right.index.values
            Y_left = Y.loc[left_idx]
            Y_right = Y.loc[right_idx]

            weighted_gini_impurity = get_gini_impurity(Y_left) * len(Y_left)/ len(Y) + get_gini_impurity(Y_right) * len(Y_right)/ len(Y)

            if weighted_gini_impurity < best_gini:
                best_gini = weighted_gini_impurity
                self.split_criterion = columnname


    def split(self, X: pd.DataFrame, Y: pd.Series):
        X_left = X.loc[X[self.split_criterion] <= self.split_value]
        X_right = X.loc[X[self.split_criterion] > self.split_value]
        Y_left = Y.loc[X_left.index]
        Y_right = Y.loc[X_right.index]

        if self.depth < self.max_depth:
            if len(X_left.index) > 1:
                self.left = DecisionNode(self.depth +1, self.max_depth)
            else:
                self.left = Leaf(self.depth + 1, self.max_depth)
            if len(X_right.index) > 1:
                self.right = DecisionNode(self.depth + 1, self.max_depth)
            else:
                self.right = Leaf(self.depth +1, self.max_depth)
        else:
            self.left = Leaf(self.depth +1, self.max_depth)
            self.right = Leaf(self.depth +1, self.max_depth)

        self.left.train(X_left, Y_left)
        self.right.train(X_right, Y_right)

    def predict(self,  X: pd.Series) -> int:
        if X[self.split_criterion] < self.split_value:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    def print(self):
        print(f"---- {self.split_criterion} ----")
        self.left.print()
        self.right.print()

class Leaf(Node):
    def __init__(self, depth, max_depth):
        self.value = None
        self.depth = depth
        self.max_depth = max_depth

    def train(self, X: pd.DataFrame, Y: pd.Series):
        if len(Y) == 0:
            self.value = 0
        else :
            self.value = Y.mode()[0]

    def predict(self, X: pd.Series) -> int:
        return self.value

    def print(self):
        print(f"---- {self.value} ----")
