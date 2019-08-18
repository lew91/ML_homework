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
        """
        Fit the regression coefficients via maximum likelihood

        Parameters
        ---------
        X : numpy array of shape '(N, M)'
            A dataset consisting of 'N' examples, each of dimension 'M'
        y : numpy array of shape '(N, K)'
            The targets for each of the 'N' examples in 'X', where each target
            has  dimension 'K'.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.dot(np.linalg.inv(X.T @ X + A), X.T)
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of
        data points

        Parameters
        ----------
        X : numpy array of shape '(Z, M)'
            A dataset consisting of 'Z' new examples, each of dimension 'M'.

        Returns
        ------
        y_pred : numpy array of shape '(Z, K)'
               The model predictions for the items in 'X'.
        """
        # convert X to design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return np.dot(X, self.beta)


class LogisticRegression:
    def  __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        A simple logistic regression model fit via gradient descent on the
        penalized negative log likelihood

        Parameters
        ----------
        penalty : {'l1', 'l2'}
             The type of regularization penalty to apply on the coefficients
            'beta'. Default is 'l2'.
        gamma : float in [0, 1]
             The regularization weight, larger values correspond to larger
             regularization penalties, and value of 0 indicates no penalty.
             Default is 0.
        fit_intercept : bool
             Whether to fit an intercept term in addition  to the coefficients in
             b. If True, the estimates for 'beta' will have 'M + 1' dimensions,
             Where the first dimension corresponds to the intercept. Default is
            True.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent on the negative
        log likelihood

        Parameters
        ----------
        X : numpy array of shape '(N, M)'
            A dataset consisting of 'N' examples, each of dimension 'M'
        y : numpy array of shape '(N, )'
            The binary targets for each of the 'N' examples in 'X'.
        lr : float
            The gradient descent learning rate. Default is 1e-7.
        max_iter : float
            The maximum number of iterations to run gradient descent
            solver. Default is 1e7.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        l_prev = np.inf
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            y_pred = sigmoid(np.dot(X, self.beta))
            loss = self._NLL(X, y, y_pred)
            if l_prev - loss < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        """
        Penalized negative log likelihood of the targets under the current
        model.

        .. math::

             \\text{NLL} = - \\frac{1}{n} \left[
               \left(\sum^N_{i=0} y_i \log \hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right) \\
             - frac{\gamma}{2} ||\mathbf{b}||_2 \\right]
        """
        N, M = X.shape
        order = 2 if self.penalty == "l2" else 1
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = 0.5 * self.gamma * np.linalg.norm(self.beta, ord=order) ** 2
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """
        Gradient of the penalized negative log likelihood wrt beta
        """
        N, M = X.shape
        p = self.penalty
        beta = self.beta
        gamma = self.gamma 
        l1nom = lambda x: np.linalg.norm(x, 1)
        d_penalty = gamma * beta if p == "l2" else gamma * l1nom(beta)  * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters
        ---------
        X : numpy array of shape '(Z, M)'
            A dataset consisting of 'Z' new examples, each of dimension 'M'.

        Returns
        ------
        y_pred : numpy array of shape '(Z, )'
            The model prediction probabilities for the items in 'X'.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))
