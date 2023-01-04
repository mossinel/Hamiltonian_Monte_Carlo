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
    #print(df.head())
    y = df["low"].copy()
    df = df.drop("Unnamed: 0", axis=1)
    df = df.drop("low", axis=1)
    df = df.drop("bwt", axis=1)
    df.rename(columns={"race": "af_am"}, inplace=True)
    df.insert(3, "Other", df["af_am"], True)
    df.rename(columns={"ftv": "visit"}, inplace=True)
    df.insert(9, "visits", df["visit"], True)
    print(df.head())

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


def autocovariance(X): # X is only the tail of q after the burn-in
    mu = np.mean(X, axis=0)
    N = len(X[:, 0])
    D = len(X[0, :])
    cov = np.zeros([N-1, D])
    
    for k in range(N-1):
        for i in range(N-k-1):
            cov[k, :] += ((X[i+k, :])-mu)*(X[i, :]-mu)
        cov[k, :] = (1/(N-k-1))*cov[k, :]
        
    return cov

def autocorrelation(autocov):   
    corr = np.zeros(np.shape(autocov))
    D = len(autocov[0, 0, :])
    for i in range(int(len(autocov[0, :, 0]))):
        for k in range(D):
            corr[:, i, k] = np.divide(autocov[:, i, k], np.maximum(np.abs(autocov[:, 0, k]), np.finfo(np.float64).eps))
    
    return corr

def get_M(autocov): # check adaptation of function for  2 dimensions
    D = len(autocov[0, :])
    M = np.zeros(D)
    for d in range(D):
        M[d] = None
        for k in range(int(len(autocov[:, d])/2)):
            if ((autocov[2*k, d] + autocov[2*k+1, d])<=0):
                M[d] = 2*k
                break
        if M[d] is None:
            M[d] = int(len(autocov[:, d])-2)
    
    
    return M

def get_sigma(autocov): # To check (difference between serie 13 and lecture notes)
    M = np.ndarray.astype(get_M(autocov), int)
    D = len(autocov[0, :])
    S = np.zeros(D)
    for d in range(D):
        id = range(1, M[d])
        S[d] = autocov[0, d] + 2*np.sum(autocov[id, d])

    return S


def effective_sample_size(autocov):
    [n, N, D] = np.shape(autocov[:, :, :])
    ess = np.zeros([n, D])
    for i in range(n):
        sigma = get_sigma(autocov[i, :, :])
        for d in range(D):
        #print("Sigma=", sigma, ", c0=", autocov[i, 0, :], ", N=", N)
            ess[i, d] = N*np.divide(autocov[i, 0, d], np.maximum(sigma[d],  np.finfo(np.float64).eps))

    return ess


if __name__ == "__main__":

    # ATTENTION: X should be modified. It is not the same as the one described in the project text
    # for example, race column is only one, not 2

    X, y = dataset_import()

    q0 = np.zeros((1, X.shape[1]))
    eps = 0.01
    m = np.ones(X.shape[1])*1
    T = 0.4
    N = 200
    sigma = 1000
    B=100

    (q, ratio) = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)

    error = np.zeros(N)
    for i in range(N):
       error[i] = log_posterior(q[i, :], X, y, sigma)

    print(np.shape(q))
    fig, ax = plt.subplots(1, 3)
    ax[0].hist(q[B:, 0])
    ax[1].hist(q[B:, 1])
    ax[2].plot(q[:, 0])
    #ax[3].plot(error)
    print(np.mean(q[B:, :], axis=0))


    plt.show()
