import numpy as np


class Perceptron:
    def __init__(self, fit_intercept=True):
        """
        A simple perceptron model fit via gradient descent

        Parameters
        ---------
        fit_intercept : bool
             Whether to fit an intercept term in addition to the coefficients
             in  b. If True, the esimates for 'beta' will have 'M + 1'
             dimensions, where the first dimension corresponds to the intercept.
             Default is True.
        """
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y, tol=1e-7, lr=0.01, max_iter=500):
        """
        Fit the regression coefficients via gradient 

        Parameter
        -------
        X : NumPy array, shape '(N, M)'
            A dataset consisting of 'N' examples, each of dimension 'M'.
        y : NumPy array, shape '(N, )'
            The binary targets for each of the 'N' examples in 'X'
        tol : float
            When to stop iteration, for the excepted errors rate (loss).
        lr : float
            The gradient learning rate. Default is 0.01
        max_iter : int
            The maximum number of iterations to run the gradient solver.
            Default is 500.
        """
        err_msg = "The dataset and the label must have same dimension"
        assert (X.shape[0] == y.shape[0]), err_msg

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        self.target_ = np.unique(y)
        assert self.target_.shape[0] == 2, \
            "label must be two classes!,but there have {}".format(self.target_.shape[0])

        self.beta = np.zeros(X.shape[1])

        self.n_iter = 0
        for _ in range(max_iter):
            y_pred = self._sign(X)
            loss = self._perceptron_loss(y, y_pred)
            if loss < tol:
                return
            self.beta += lr * self._grad(X, y, y_pred)
            self.n_iter += 1

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
        """
        The binary targets whether is {-1, +1} or {0, 1}
        """
        return np.where(np.dot(x, self.beta) >= 0, max(self.target_), min(self.target_))

    def _perceptron_loss(self, y, y_pred):
        N = y.shape[0]
        error = y - y_pred
        return np.linalg.norm(error, ord=1) / N

    def _grad(self, X, y, y_pred):
        N = X.shape[0]
        return  np.dot(y - y_pred, X) / N

    def predict(self, X):
        if self.fit_intercept:
            if X.ndim == 1:
                X = np.r_[1, X]
            else:
                X = np.c_[np.ones(X.shape[0]), X]

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

