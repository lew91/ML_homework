import numpy as np
from numpy.testing import assert_allclose


def _log_gaussian_pdf(x_i, mu, sigma):
    """
    Compute log(x_i | mu, sigma)
    """
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)

    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)


def _logsumexp(log_probs, axis=None):
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)


class GaussianMixture:
    """
    Gaussian Mixture.
    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Via the Expectation maximization

    Parameters
    ----------
    C : int (default: 3)
       The number of clusters or mixture components in the Gaussian Mixture
    seed : (default: None)
       Seed for the random initialize parameters generate.


    Attributes
    ----------
    pi : numpy array, shape `(n_components, )`
         The weights of each mixture component

    mu : numpy array, shape `(n_components, n_features)`
         The mean of each mixture component.
    sigma : numpy array,
         The covariance of each mixture componet.
    elbo : float
         The Likelihood lower bound
    n_iter : int
         The iter number of the Gaussian Mixture process 
    """
    def __init__(self, C=3, seed=None):
        """Gaussian Mixture initialize.

        Parameters
        ----------
        C : int (default: 3)
           The number of clusters or mixture components in the Gaussian Mixture
        seed : (default: None)
           Seed for the random initialize parameters generate.

        """
        self.C = C  # number of clusters or mixture components
        self.N = None  # number of samples
        self.d = None  # dimension of each features

        if seed:
            np.random.seed(seed)

    def _initialize_params(self):
        """
        Random initialize the starting parameters
        """
        C, d = self.C, self.d
        rr = np.random.randn(C)

        self.pi = rr / rr.sum()  # cluster priors
        self.Q = np.zeros((self.N, C))  # variations distribution q(T)
        self.mu = np.random.uniform(-5, 10, C * d).reshape(C,
                                                           d)  # cluster means
        self.sigma = np.array([np.identity(d)
                               for _ in range(C)])  # cluster covariances

        self.elbo = -np.inf

    def _likelihood_lower_bound(self):
        """
        Compute the likelihood lower bound under current parametes
        """
        N = self.N
        C = self.C

        eps = np.finfo(float).eps
        expec1, expec2 = 0.0, 0.0
        for i in range(N):
            x_i = self.X[i]

            for c in range(C):
                pi_k = self.pi[c]
                z_nk = self.Q[i, c]
                mu_k = self.mu[c, :]
                sigma_k = self.sigma[c, :, :]

                log_pi_k = np.log(pi_k + eps)
                log_p_x_i = _log_gaussian_pdf(x_i, mu_k, sigma_k)
                prob = z_nk * (log_pi_k + log_p_x_i)

                expec1 += prob
                expec2 += z_nk * np.log(z_nk + eps)

        loss = expec1 - expec2
        return loss

    def fit(self, X, max_iter=100, tol=1e-3, verbose=False):
        """
        Parameters
        ---------
        X : numpy array of shape (N, d)
            A collection of 'N' training data points, each with dimension 'd'
        
        max_iter : int (default: 100)
            The maximum number of EM updates to perform before terminating training
        
        tol : float (default: 1e-3)
            The convergence tolerance. Training is terminated if the difference in
            VLB between the current and previous iteration is less than 'tol'
        
        verbose : bool (default: False)
            Whether to print the VLB at each training iteration.

        Returns
        --------
        If Sequence return 0 , if not then return -1
            Whether training terminated without incident (0) or one of the mixture
        components collapsed and training was halted prematurely (-1)
        """
        self.X = X
        self.N = X.shape[0]  # number of objects
        self.d = X.shape[1]  # dimension of each object

        self._initialize_params()
        prev_vlb = -np.inf

        for _iter in range(max_iter):
            try:
                self._E_step()
                self._M_step()
                vlb = self._likelihood_lower_bound()

                if verbose:
                    print("{}.lower bound: {}".format(_iter + 1, vlb))

                converged = _iter > 0 and np.abs(vlb - prev_vlb) <= tol
                if np.isnan(vlb) or converged:
                    break

                prev_vlb = vlb

                # retain best parameters across fits
                if vlb > self.elbo:
                    self.elbo = vlb

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                return -1

            self.n_iter = _iter
        return 0

    def _E_step(self):
        for i in range(self.N):
            x_i = self.X[i, :]
            q_i = self._prod_q_i(x_i, self.pi, self.mu, self.sigma)
            assert_allclose(np.sum(q_i, axis=0),
                            1,
                            err_msg="{}".format(np.sum(q_i)))
            self.Q[i, :] = q_i

    def _prod_q_i(self, x_i, pi, mu, sigma):
        """calculate each cluster probability"""
        denom_vals = []
        for c in range(self.C):
            pi_c = self.pi[c]
            mu_c = self.mu[c, :]
            sigma_c = self.sigma[c, :, :]

            log_pi_c = np.log(pi_c)
            log_p_x_i = _log_gaussian_pdf(x_i, mu_c, sigma_c)

            denom_vals.append(log_p_x_i + log_pi_c)

        log_denom = _logsumexp(denom_vals)
        return np.exp([num - log_denom for num in denom_vals])

    def _M_step(self):
        C, N, X = self.C, self.N, self.X
        denoms = np.sum(self.Q, axis=0)

        # update cluster priors
        self.pi = denoms / N

        # update cluster means
        nums_mu = [np.dot(self.Q[:, c], X) for c in range(C)]
        for ix, (num, den) in enumerate(zip(nums_mu, denoms)):
            self.mu[ix, :] = num / den if den > 0 else np.zeros_like(num)

        # update cluster covariances
        for c in range(C):
            mu_c = self.mu[c, :]
            n_c = denoms[c]

            outer = np.zeros((self.d, self.d))
            for i in range(N):
                wic = self.Q[i, c]
                xi = self.X[i, :]
                outer += wic * np.outer(xi - mu_c, xi - mu_c)

            outer = outer / n_c if n_c > 0 else outer
            self.sigma[c, :, :] = outer

        assert_allclose(np.sum(self.pi),
                        1,
                        err_msg="{}".format(np.sum(self.pi)))

    def predict_proba(self, X):
        if X.ndim == 1:
            return self._prod_q_i(X, self.pi, self.mu, self.sigma)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.C))

        for i in range(n_samples):
            x_i = X[i, :]

            proba[i, :] = self._prod_q_i(x_i, self.best_pi, self.best_mu,
                                         self.sigma)
        return proba

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)
