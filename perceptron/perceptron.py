import numpy as np


class Perceptron:
    def __init__(self, fit_intercept=True):
        self.beta = None
        self.n_class = None 
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, max_iter=500):
        err_msg = "The dataset and the label must have same dimension"
        assert (X.shape[0] == y.shape[0]), err_msg

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        self.n_class = list(set(y))
        assert len(self.n_class) == 2, \
            "label must be two classes!,but there have {}".format(len(self.n_class))
        
        #self.beta = np.random.rand(X.shape[1])
        self.beta = np.zeros(X.shape[1])

        for _ in range(max_iter):
            for x, est in zip(X, y):
                y_pred = self._sign(x)
                error = est - y_pred
                self.beta += lr * error * x
            
    def score(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        y_pred = self._sign(X)
        acc_ = 0.0
        for i in range(len(y)):
            if y_pred[i] == y[i]:
                acc_ += 1
        return acc_ / len(y)
            
    def _sign(self, x):
        return np.where(np.dot(x, self.beta) >= 0, max(self.n_class), min(self.n_class))

    def _grad(self, X, y, y_pred):
        N, M = X.shape
        return  np.dot(y - y_pred, X) / N

    def predict(self, X):
        if self.fit_intercept:
            if X.ndim == 1:
                X = np.r_[1, X]
            else:
                X = np.c_[np.ones(X.shape[0]), X]

        #return np.sign(np.dot(X,self.beta))
        return self._sign(X)



# test predictions
def create_dataset():
    dataset = [
        [2.7810836, 2.550537003, -1],
        [1.465489372, 2.362125076, -1],
        [3.396561688, 4.400293529, -1],
        [1.38807019, 1.850220317, -1],
        [3.06407232, 3.005305973, -1],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1],
    ]
    data = np.array(dataset)
    return data[:,:-1], data[:,-1]

