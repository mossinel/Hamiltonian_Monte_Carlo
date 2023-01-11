import seaborn as sns
import pandas as pd
import os
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np


def log_posterior(q, X, y, sigma=1000):  # log posterior

    n, p = (X.shape[0], X.shape[1] - 1)

    return q @ X.T @ (y.T - np.ones(n)) - np.ones(n) @ (
        np.log1p(np.exp(-X @ q)) - q @ q.T / (2 * sigma)
    )


def grad_log(X, y, q, sigma):  # gradient of the log posterior
    value = np.exp(-X @ q)
    place = np.where(value == np.inf)[0]

    value = value / (1 + value)
    value[place] = 1.0

    grad = X.T @ (y - np.ones(y.shape[0]) + value) - q / sigma

    return grad


def K(p, m):  # ""kinetic energy"
    return np.sum(1 / 2 * np.divide(p * p, m))


def Verlet(q0, m, p0, eps, T, X, y, sigma):  # Verlet method
    t = 0
    q = q0
    p = p0
    while t < T:
        p_tmp = p + eps / 2 * grad_log(X, y, q, sigma)

        q = q + eps * np.divide(p_tmp, m)

        p = p_tmp + eps / 2 * grad_log(X, y, q, sigma)

        t = t + eps
    return q, p


def Hamiltonian_Monte_Carlo(
    q0, m, N, T, eps, X, y, sigma
):  # HMC scheme adapted for the low-weight problem

    q = np.zeros([N + 1, q0.shape[1]])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=np.sqrt(m))

        q_star, p_star = Verlet(
            q0=q[i, :], m=m, p0=p, eps=eps, T=T, X=X, y=y, sigma=sigma
        )

        u = np.random.uniform()
        if u < np.exp(
            log_posterior(q_star, X, y, sigma)
            - log_posterior(q[i, :], X, y, sigma)
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



def MH_one_at_a_time(q0, N, var, X, y, sigma): # one variable at a time Metropolis Hastings
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


def dataset_import():
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "src\\birthwt.csv")
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
    # normalization of age and lwt, only non categorical variables
    X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])
    X[:, 2] = (X[:, 2] - np.mean(X[:, 2])) / np.std(X[:, 2])
    return X, y


def autocovariance(X): # compute the autocovariance of the chain
    # X is only the tail of q after the burn-in
    mu = np.mean(X, axis=0)
    N = len(X[:, 0])
    D = len(X[0, :])
    cov = np.zeros([N-1, D])
    
    for k in range(N-1):
        for i in range(N-k-1):
            cov[k, :] += ((X[i+k, :])-mu)*(X[i, :]-mu)
        cov[k, :] = (1/(N-k-1))*cov[k, :]
        
    return cov

def autocorrelation(autocov): # compute the autocorrelation of the chain from its autocovariance    
    corr = np.zeros(np.shape(autocov))
    D = len(autocov[0, 0, :])
    for i in range(int(len(autocov[0, :, 0]))):
        for k in range(D):
            corr[:, i, k] = np.divide(autocov[:, i, k], np.maximum(np.abs(autocov[:, 0, k]), np.finfo(np.float64).eps))
    
    return corr

def get_M(autocov): # compute the M value for each parameters of q
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

def get_sigma(autocov): # compute the time-average variance constant of the Markov chain from its autocovariance
    M = np.ndarray.astype(get_M(autocov), int)
    D = len(autocov[0, :])
    S = np.zeros(D)
    for d in range(D):
        id = range(1, M[d])
        S[d] = autocov[0, d] + 2*np.sum(autocov[id, d])

    return S


def effective_sample_size(autocov): # compute the effective sample size of the Markov chain from its autocovariance
    [n, N, D] = np.shape(autocov[:, :, :])
    ess = np.zeros([n, D])
    for i in range(n):
        sigma = get_sigma(autocov[i, :, :])
        for d in range(D):
        #print("Sigma=", sigma, ", c0=", autocov[i, 0, :], ", N=", N)
            ess[i, d] = N*np.divide(autocov[i, 0, d], np.maximum(sigma[d],  np.finfo(np.float64).eps))

    return ess


if __name__ == "__main__":

    X, y = dataset_import()

    q0 = np.zeros((1, X.shape[1]))
    eps = 0.01
    m = np.ones(X.shape[1])
    T = 0.40
    N = 6000
    sigma = 1000
    B = 4800

    (q, ratio) = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)

    # histograms
    sns.set_style("darkgrid")
    rel = sns.histplot(
        data=q[:, 0],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("intercept", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 1],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("age", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 2],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("lwt", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 3],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("race2black", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 4],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("race2other", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 5],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("smoke", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 6],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("ptd", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 7],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("ht", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 8],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("ui", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 9],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("ftv21", fontsize=20)
    plt.show()

    rel = sns.histplot(
        data=q[:, 10],
        kde=True,
        stat="probability",
        color="r",
        label="probabilities",
    )
    plt.title("ftv22+", fontsize=20)
    plt.show()
