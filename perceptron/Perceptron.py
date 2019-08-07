from numpy import *

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit (self, X, y):
        self.w_ = zeros(X.shape[1] + 1)
        self.errors_ = []

        for _ in range(self.n_iter):
            self.errors_ = 0
            for xi, est in zip(X, y):
                est = self.eta * (est - self.predict(xi))
                self.w_[0] += est   # bias, b <- b + \eta * y_i
                self.w_[1:] += est * xi   # weights, w <- w + \eta * y_i * x_i
                errors += int(est != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return where(self.net_input(X) >=0.0, 1, -1)



