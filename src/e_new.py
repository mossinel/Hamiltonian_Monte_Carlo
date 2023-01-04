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
        - np.ones(n) @ (np.log(1 / logistic(X @ q)))
        - q.T @ q / (2 * sigma)
    )


def grad_log(X, y, q, sigma):
    grad = (
        X.T @ (y - np.ones(y.shape[0]))
        + X.T @ (np.exp(-X @ q) / (logistic(X @ q)))
        - q / sigma
    )
    return grad


def K(p, m):
    return np.sum(1 / 2 * np.divide(p * p, m))


def Verlet(q0, m, p0, eps, T, X, y, sigma):
    t = 0
    q = q0
    p = p0
    while t < T:
        p_tmp = p + eps / 2 * grad_log(X, y, q, sigma)
        # print(p_tmp)
        q = q + eps * np.divide(p_tmp, m)
        # print("--")
        # print(q)
        p = p_tmp + eps / 2 * grad_log(X, y, q, sigma)
        # print(grad_log(X, y, q, sigma))
        # print("--")
        # print(p)
        t = t + eps
    return q, p


def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma):
    # taken from "d.py"
    q = np.zeros([N + 1, q0.shape[1]])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        # p = st.norm.rvs(loc=0, scale=np.sqrt(np.ones(q0.shape[0]) * np.sqrt(sigma)))
        p = st.norm.rvs(loc=0, scale=np.sqrt(m))
        # print(m)

        q_star, p_star = Verlet(
            q0=q[i, :], m=m, p0=p, eps=eps, T=T, X=X, y=y, sigma=sigma
        )
        # print(q_star)
        # print(p_star)
        u = np.random.uniform()
        # print(-log_posterior(q_star, X, y, sigma))
        # print(+log_posterior(q[i, :], X, y, sigma))
        # print(-K(p_star, m))
        # print(+K(p, m))
        if u < np.exp(
            -log_posterior(q_star, X, y, sigma)
            + log_posterior(q[i, :], X, y, sigma)
            - K(p_star, m)
            + K(p, m)
        ):
            q[i + 1, :] = q_star
            accepted = accepted + 1
        else:
            q[i + 1, :] = q[i, :]
            rejected = rejected + 1
            print(f"Problem:{i+1}")
    ratio = accepted / (accepted + rejected)
    return q, ratio


def dataset_import():
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "\\Hamiltonian_Monte_Carlo\\src\\birthwt.csv")
    df = pd.read_csv(data_dir)
    y = df["low"].copy()
    df = df.drop("Unnamed: 0", axis=1)
    df = df.drop("low", axis=1)
    df = df.drop("bwt", axis=1)
    df.rename(columns={"race": "af_am"}, inplace=True)
    df.insert(3, "Other", df["af_am"], True)
    df.rename(columns={"ftv": "visit"}, inplace=True)
    df.insert(9, "visits", df["visit"], True)

    # transform to array
    X = df.to_numpy()
    y = y.to_numpy()
    # visits last months
    X[np.where(X[:, -1] < 2)[0], -1] = 0
    X[np.where(X[:, -1] >= 2)[0], -1] = 1
    # only one visit
    X[np.where(X[:, -2] != 1)[0], -2] = 0
    # afro american
    X[np.where(X[:, 2] != 2)[0], 2] = 0
    X[np.where(X[:, 2] == 2)[0], 2] = 1
    # other
    X[np.where(X[:, 3] != 3)[0], 3] = 0
    X[np.where(X[:, 3] == 3)[0], 3] = 1

    # intersept
    X = np.c_[np.ones((X.shape[0], 1)), X]

    return X, y


if __name__ == "__main__":

    # ATTENTION: X should be modified. It is not the same as the one described in the project text
    # for example, race column is only one, not 2

    X, y = dataset_import()

    q0 = np.ones((1, X.shape[1]))
    eps = 0.01
    m = np.ones(X.shape[1])
    T = 0.10
    N = 80
    sigma = 1000

    (q, ratio) = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)
