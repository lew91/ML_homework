import numpy as np

class LinearRegression:
    """
    The simple linear regression model is
    
    y = bX + e where e ~ N(0, sigma^2 * I)

    In probabilistic terms this corresponds to
    
    y - bX ~ N(0, sigma^2 * I)
    y | x, b ~ (bX, sigma^2 * I)

    The loss for the model is simply the squared error between the model
    predictions and the true values:

    Loss = ||y - bX||^2

    The MLE for the model parameters b can be computed in closed form via
    the normal equation:

    b = (X^T X)^{-1} X^T y

    where (X^T X)^{-1} X^T is known as the pseudoinverse / Moore-penrose
    inverse.
    """
    def __init__(self, fit_intercept=True):
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.beta = np.dot(pseudo_inverse, y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return np.dot(X, self.beta)

    
class RidgeRegression:
    """
     Ridge regression uses the same simple linear regression model but adds an
    additional penalty on the L2-norm of the coefficients to the loss function.
    This is sometimes known as Tikhonov regularization.

    In particular, the ridge model is still simply

        y = bX + e  where e ~ N(0, sigma^2 * I)

    except now the error for the model is calcualted as

        RidgeLoss = ||y - bX||^2 + alpha * ||b||^2

    The MLE for the model parameters b can be computed in closed form via the
    adjusted normal equation:

        b = (X^T X + alpha I)^{-1} X^T y

    where (X^T X + alpha I)^{-1} X^T is the pseudoinverse / Moore-Penrose
    inverse adjusted for the L2 penalty on the model coefficients.
    """
    def __init__(self, alpha=1, fit_intercept=True):
        """
        A ridge regression model fit via the normal equation.

        Parameters
        ----------
        alpha : float(default: 1)
        L2 regularization coefficient. Higher values correspond to larger
        penalty on the l2 norm of the model coefficients

        fit_intercept : bool (default: True)
        Whether to fit an additional intercept term in addition to the
        model coefficients
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.dot(np.linalg.inv(X.T @ X + A), X.T)
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return np.dot(X, self.beta)

    
