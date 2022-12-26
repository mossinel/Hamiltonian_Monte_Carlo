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
        q.T * X.T * (y - np.ones(n))
        - np.ones((n, 1)) * (np.log(1 / logistic))
        - q.T * q / (2 * sigma * sigma)
    )


def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha):
    # taken from "d.py"
    q = np.zeros([N + 1, 2])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=np.sqrt(m))
        q_star, p_star = Verlet(q[i, :], p, T, eps, alpha, m)
        u = np.random.uniform()
        if u < np.exp(
            -U(q_star, alpha) + U(q[i, :], alpha) - K(p_star, m) + K(p, m)
        ):
            q[i + 1, :] = q_star
            accepted = accepted + 1
        else:
            q[i + 1, :] = q[i, :]
            rejected = rejected + 1
            # print("Problem")
    ratio = accepted / (accepted + rejected)
    return q, ratio


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "src\\birthwt.csv")
    df = pd.read_csv(data_dir)

    sigma = 1000
