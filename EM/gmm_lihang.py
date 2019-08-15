import numpy as np
import numpy.testing import assert_allclose


def Gaussian_(X, mu, Sigma):
    """
    Redefine Gaussian density function
    f(X) = frac{1}{(2pi)^{d/2} * |Sigma|^{1/2}} * exp (- (X-mu)'Sigma^{-1}(x-mu)/2 )
    """
    d = np.shape(Sigma)[0]
    
    try:
        det = np.linalg.det(Sigma)
        s_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        print("singular matrix: components collapsed")

    x_diff = (X - mu).reshape((1, d))

    prob = 1.0 /(np.power(np.power(2 * np.pi, d) * np.abs(det), 0.5)) * \
        np.exp(-0.5 * x_diff.dot(s_inv).dot(x_diff.T) )[0][0]

    return prob


def logsumexp(log_probs, axis=None):
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)


class GMM_Lihang:
    def __init__(self, C=3, seed=None):
        """
        Gaussian mixture model trained via the expectation maximization

        Parameters
        ----------
        C : int (default: 3)
            The number of clusters / mixture components in the GMM
        seed : int (default: None)
            Seed for the random number generator
        """
        self.C = C # number of clusters
        self.N = None  # number of objects
        self.d = None  # dimension of each object

        if seed:
            no.random.seed(seed)

    def _initialize_params(self):
        """
        Randomly initialize the starting GMM parameters
        """
        C, d = self.C, self.d
        rr = np.random.rand(C)

        self.alpha = rr / rr.sum()  # weights of clusters priors
        self.mu = np.random.uniform(-5, 10 , C * d).reshape(C, d) # cluster means
        self.sigma = np.array([np.identity(d) for _ in range(C)])  #  cluster covariances

        self.gammas = np.zeros((self.N, C))

        self.best_alpha = None
        self.best_mu = None
        self.best_sigma = None
        self.best_elbo = -np.inf  

    def likelihood_lower_bound(self):
        """
        Compute the LLB under the current GMM parametes
        """
        N = self.N
        C = self.C

        eps = np.finfo(float).eps
        expec1, expec2 = 0.0, 0.0
        for n in range(N):
            x_i = self.X[i]
            
            for c in range(C):
                alpha_k = self.alpha[c]
                z_nk = self.gammas[n, c]
                mu_k = self.mu[c, :]
                sigma_k = self.sigma[c, :, :]

                prob = z_nk * (np.log(alpha_k) + np.log(Gaussian_(x_i, mu_k, sigma_k)))

                expec1 += prob
                expec2 += z_nk * np.log(z_nk + eps)

        loss = expec1 - expec2
        return loss

    def fit(self, X, max_iter=100, tol=1e-3, verbose=False):
        """
        Fit the parameters of the GMM on some training data
        """
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[1]

        self._initialize_params()
        prev_vlb = -np.inf

        for _iter in range(max_iter):
            try:
                self._E_step()
                self._M_step()
                vlb = self.likelihood_lower_bound()

                if verbose:
                    print("{}. Lower bound: {}".format(_iter + 1, vlb))

                converged = _iter > 0 and np.abs(vlb - pre_vld) <= tol
                if np.isnan(vlb) or converged:
                    break

                prev_vlb = vlb

                #  retain best parameters across fits
                if vlb > self.best_elbo:
                    self.best_elbo = vlb
                    self.best_alpha = self.alpha
                    self.best_mu = self.mu
                    self.best_sigma = self.sigma

            except np.linalg.LinAlgError:
                print("singular  matrix: components collapsed")
                return -1
        return 0

    def _E_step(self):
        for n in range(self.N):
            x_n = self.X[n, :]

            denom_vals = []
            for c in range(self.C):
                a_c = self.alpha[c]
                mu_c = self.mu[c, :]
                sigma_c = self.sigma[c, :, :]
                
                posterior = np.log(a_c) + np.log(Gaussian_(x_n, mu_c, sigma_c))
                denom_vals.append(posterior)

            log_denom = logsumexp(denom_vals)
            gamma = np.exp([num - log_denom for num in denom_vals])
            assert_allclose(np.sum(gammas), 1, err_msg="{}".format(np.sum(gammas)))
            
            self.gammas[n, :] = gamma

    def _M_step(self):
        C, N, X = self.C, self.N, self.X
        denoms = np.sum(self.gammas, axis=0)

        # update cluster priors
        self.alpha = denoms / N

        # update cluster means
        nums_mu = [np.dot(self.gammas[:, c], X) for c in range(C)]
        for ix, (num, den) in enumerate(zip(nums_mu, denoms)):
            self.mu[ix, :] = num / den if den > 0 else np.zeros_like(num)

        # update cluster covariances
        for c in range(C):
            mu_c = self.mu[c, :]
            n_c = denoms[c]

            outer = np.zeros((self.d, self.d))
            for i in range(N):
                wic = self.gammas[i, c]
                xi = self.X[i, :]
                outer += wic * np.outer(xi - mu_c, xi - mu_c)

            outer = outer / n_c if n_c > 0 else outer
            self.sigma[c, :, :] = outer

        assert_allclose(np.sum(self.alpha), 1, err_msg="{}".format(np.sum(self.alpha)))
        
