import pandas as pd
import os
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

# https://cran.r-project.org/web/packages/hmclearn/vignettes/logistic_regression_hmclearn.html

logistic = lambda z: 1 / (1 + np.exp(-z))


def prior(p, sigma=1000):
    """
    gaussian of variance sigma I of shape p+1 x p+1
    """
    return np.random.standard_normal(p + 1) * np.sqrt(sigma)


def log_posterior(q, X, y, sigma=1000):
    """
    q : column vector
    y : label
    X : matrix n x (p+1)
    sigma : variance (scalar)
    """

    n, p = (X.shape[0], X.shape[1] - 1)

    return (
        q @ X.T @ (y.T - np.ones(n))
        - np.ones((n, 1)) * (np.log(1 / logistic(-X @ q)))
        - q.T @ q / (2 * sigma * sigma)
    )


def dU(X, y, q, sigma):
    grad = (
        X.T @ (y.T - np.ones(y.shape[0]))
        + X.T @ (np.exp(-X @ q) / (logistic(-X @ q)))
        - q / sigma
    )
    return grad


def K(p, sigma=1000):
    return np.sum(1 / 2 * np.divide(p * p, sigma))


def Verlet(q0, p0, eps, T, m, X, y, sigma):
    t = 0
    q = q0
    p = p0
    while t < T:
        p_tmp = p - eps / 2 * dU(X, y, q, sigma)
        q = q + eps * np.divide(p_tmp, m)
        p = p_tmp - eps / 2 * dU(X, y, q, sigma)
        t = t + eps
    return q, p


def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma):
    # taken from "d.py"
    q = np.zeros([q0.shape[0], N + 1])
    q[:, 0] = q0[:, 0]
    accepted = 0
    rejected = 0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=np.sqrt(sigma))
        print(p)
        q_star, p_star = Verlet(q[:, i], p, T, eps, m, X, y, sigma)
        print(p_star)
        u = np.random.uniform()
        print((-log_posterior(q_star, X, y, sigma)).shape)
        print(+log_posterior(q[:, i], X, y, sigma))
        print(K(p_star, sigma))
        # print(+K(prior(q0.shape[0], sigma), sigma))
        if u < np.exp(
            -log_posterior(q_star, X, y, sigma)
            + log_posterior(q[:, i], X, y, sigma)
            - K(p_star, sigma)
            + K(prior(q0.shape[0], sigma), sigma)
        ):
            q[:, i + 1] = q_star
            accepted = accepted + 1
        else:
            q[:, i + 1] = q[:, i]
            rejected = rejected + 1
            # print("Problem")
    ratio = accepted / (accepted + rejected)
    return q, ratio


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "src\\birthwt.csv")
    df = pd.read_csv(data_dir)
    X = df.to_numpy()
    # ATTENTION: X should be modified. It is not the same as the one described in the project text
    # for example, race column is only one, not 2
    y = X[:, 1]
    X = np.c_[np.ones((X.shape[0], 1)), X[:, 2:-1]]

    q0 = np.zeros((X.shape[1], 1))
    eps = 0.01
    m = np.ones(X.shape[1])
    T = 0.10
    N = 80
    sigma = 1000

    (q, ratio) = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)
