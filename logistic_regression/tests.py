import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import LogisticRegression as lr


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    return data[:, :2], data[:, -1]


def data_mat(x):
    data_m = []
    for d in x:
        data_m.append([1.0, *d])
    return data_m


def score(X_test, y_test, clf):
    correct = 0
    X_test = data_mat(X_test)
    for x, y in zip(X_test, y_test):
        result = np.dot(x, clf.beta)
        if (result > 0 and y ==1) or (result <0 and y==0):
            correct += 1
    return correct / len(X_test)


def plotshow(x, clf):
    x_points = np.arange(4,8)
    y_ = -(clf.beta[1] * x_points + clf.beta[0]) / clf.beta[2]
    plt.plot(x_points, y_)
    
    plt.scatter(x[:50,0], x[:50,1], label='0')
    plt.scatter(x[50:,0], x[50:,1], label='1')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lr_clf = lr.LogisticRegression()
    lr_clf.fit(X_train, y_train)

    score = score(X_test, y_test, lr_clf)
    print(score)

    plotshow(X, lr_clf)
