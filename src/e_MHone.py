import pandas as pd
import os
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

# CODE WITH SPLITTED CONTINUOUS AND CATEGORICAL FEATURES
logistic = lambda z: 1 / (1 + np.exp(-z))


def prior(p, sigma=1000):
    """
    gaussian of variance sigma I of shape p+1 x p+1
    """
    return np.random.standard_normal(p + 1) * np.sqrt(sigma)


def log_posterior(q, X, y, sigma=1000):
    """
    q : row vector
    y : label
    X : matrix n x (p+1)
    sigma : variance (scalar)
    """

    n, p = (X.shape[0], X.shape[1] - 1)

    return q @ X.T @ (y.T - np.ones(n)) - np.ones(n) @ (
        np.log1p(np.exp(-X @ q)) - q @ q.T / (2 * sigma)
    )


def grad_log(X, y, q, sigma):
    value = np.exp(-X @ q)
    place = np.where(value == np.inf)[0]
    #print(value)
    value = value / (1 + value)
    value[place] = 1.0

    grad = X.T @ (y - np.ones(y.shape[0]) + value) - q / sigma
    #print("----------")
    #print(value)
    return grad


def K(p, m):
    return np.sum(1 / 2 * np.divide(p * p, m))


def Verlet(q0, m, p0, eps, T, X, y, sigma):
    t = 0
    q = q0
    p = p0
    while t < T:
        p_tmp = p + eps / 2 * grad_log(X, y, q, sigma)
        #print(p_tmp)
        q = q + eps * np.divide(p_tmp, m)
        #print("--")
        #print(q)
        p = p_tmp + eps / 2 * grad_log(X, y, q, sigma)
        #print(grad_log(X, y, q, sigma))
        #print("--")
        #print(p)
        t = t + eps
    return q, p


def MH_one_at_a_time(q0, N, var, X, y, sigma):
    n = len(q0[0, :])
    print(n)
    q = np.zeros([N + 1, n])

    q[0, :] = q0
    accepted = 0
    rejected = 0

    for i in range(N):
        j = np.random.randint(0, n)
        q_star = np.copy(q[i, :])
        # define posterior wrt other variates (or.....use the prior, like MH?)
        q_star[j] = np.random.normal(loc=q[i, j], scale=var)
        a = np.exp(log_posterior(q_star, X, y, sigma)-log_posterior(q[i, :], X, y, sigma))
        u = np.random.uniform()
        if (u<a):
            q[i+1, :] = q_star
            accepted = accepted+1
        else:
            q[i+1, :] = q[i, :]
            rejected = rejected+1

    ratio = accepted/(accepted+rejected)
    return q, ratio



def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma):
    # taken from "d.py"
    q = np.zeros([N + 1, q0.shape[1]])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        # p = st.norm.rvs(loc=0, scale=np.sqrt(np.ones(q0.shape[0]) * np.sqrt(sigma)))
        p = st.norm.rvs(loc=0, scale=np.sqrt(m))

        q_star, p_star = Verlet(
            q0=q[i, :], m=m, p0=p, eps=eps, T=T, X=X, y=y, sigma=sigma
        )

        u = np.random.uniform()

        if u < np.exp(
            log_posterior(q_star, X, y, sigma)
            - log_posterior(
                q[i, :], X, y, sigma
            )  # signs?? see page 12 MC on R
            - K(p_star, m)
            + K(p, m)
        ):
            q[i + 1, :] = q_star
            accepted = accepted + 1
        else:
            q[i + 1, :] = q[i, :]
            rejected = rejected + 1
            #print(f"Problem:{i+1}")
    ratio = accepted / (accepted + rejected)
    return q, ratio


def dataset_import():
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "Hamiltonian_Monte_Carlo\\src\\birthwt.csv")
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
    # normalization
    X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])
    X[:, 2] = (X[:, 2] - np.mean(X[:, 2])) / np.std(X[:, 2])

    return X, y


if __name__ == "__main__":

    X, y = dataset_import()

    q0 = np.zeros((1, X.shape[1]))
    # eps_continuous = 0.1
    # eps_categorical = 0.05
    # eps = [eps_continuous, eps_categorical]
    m = np.ones(X.shape[1])
    
    N = 100000
    var = 1
    B = 10000

    sigma = 1000

    q, ratio = MH_one_at_a_time(q0, N, var, X, y, sigma)

    print(ratio)
    fig, ax = plt.subplots(1, 3)
    ax[0].hist(q[B:, 0])
    ax[1].hist(q[B:, 1])
    ax[2].plot(q[:, 0])
    print(np.mean(q[B:, :], axis=0))


    plt.show()