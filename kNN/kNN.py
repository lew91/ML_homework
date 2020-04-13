import math
import operator
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



def distance(a, b, p=2):
    """
    Calculate the distance between vector between x and y

    Parameters
    ----------
    a : 1-D vector
    b : 1-D vector
    p : the distance which calculate (default: 2)
        p = 1 manhad distance
        p = 2 euclidean distance
        p = inf  minkowski-distance
    """
    err_msg = "the 'a' and 'b' must have the same length and the length big than 1 "
    assert len(a) == len(b) and len(a) > 1, err_msg
    sum = 0
    for i in range(len(a)):
        sum += math.pow(abs(a[i] - b[i]), p)
    return math.pow(sum, 1 / p)


class kNN:
    def __init__(self, n_neighbors=3, p=2):
        self._n = n_neighbors
        self._p = p

    def fit(self, X_train, y_train):
        self._X = X_train
        self._y = y_train
        

    def predict(self, X):
        pass


class KdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right

class KdTree:
    def __init__(self, data):
        k = len(data[0])

        def createNode(split, dataset):
            if not dataset:
                return None

            dataset.sort(key=lambda x: x[split])
            split_pos = len(dataset) // 2
            median = dataset[split_pos]
            split_next = (split + 1) % k

            return KdNode(median, split,
                          createNode(split_next, dataset[: split_pos]),
                          createNode(split_next, dataset[split_pos + 1 : ]))
        self.root = createNode(0, data)

