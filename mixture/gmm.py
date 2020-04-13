import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

DEBUG = True


def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov_k)
    return norm.pdf(Y)


# E-Step
def getExpection(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]

    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    gamma = np.mat(np.zeros((N, K)))

    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, K] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in ragne(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma
