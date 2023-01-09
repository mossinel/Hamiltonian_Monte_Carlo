import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def f(q, alpha): # unnormalized distribution
    return np.exp(-alpha*(q@q-1/4)**2)

def U(q, alpha): #-log(f) (potential energy)
    return alpha*(q@q-1/4)**2

def K(p, m): #kinetic energy
    return np.sum(1/2*np.divide(p*p, m))

def dU(q, alpha): #gradient of U
    dU1 = 4*q[0]*alpha*(q@q-1/4)
    dU2 = 4*q[1]*alpha*(q@q-1/4)
    return np.asarray([dU1, dU2])


def Verlet(q0, p0, eps, T, alpha, m): #Verlet's scheme 
    t=0
    q=q0
    p=p0
    while t<T:
        p_tmp = p - eps/2*dU(q, alpha)
        q = q + eps*np.divide(p_tmp, m)
        p = p_tmp - eps/2*dU(q, alpha)
        t=t+eps
    return q, p

def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha):
    q = np.zeros([N+1, 2])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=np.sqrt(m))
        q_star, p_star = Verlet(q[i, :], p, eps, T, alpha, m)
        u = np.random.uniform()
        if (u<np.exp(-U(q_star, alpha)+U(q[i, :], alpha)-K(p_star, m)+K(p, m))):
            q[i+1, :] = q_star
            accepted = accepted+1
        else:
            q[i+1, :] = q[i, :]
            rejected = rejected+1
    ratio = accepted/(accepted+rejected)
    return q, ratio

def Metropolis_Hastings(q0, N, alpha, sigma):
    q = np.zeros([N+1, 2])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        q_star = np.random.multivariate_normal(mean=q[i, :], cov=sigma*np.eye(2))
        a = f(q_star, alpha)/np.maximum(f(q[i, :], alpha), np.finfo(np.float64).eps)
        u = np.random.uniform()
        if (u<a):
            q[i+1, :] = q_star
            accepted = accepted+1
        else:
            q[i+1, :] = q[i, :]
            rejected = rejected+1
    ratio = accepted/(accepted+rejected)
    return q, ratio


def autocovariance(X): # Autocovariance of a Markov chain X, X tail of q after burn-in lag B
    mu = np.mean(X, axis=0)
    N = len(X[:, 0])
    cov = np.zeros([N-1, 2])
    
    for k in range(N-1):
        for i in range(N-k-1):
            cov[k, :] += ((X[i+k, :])-mu)*(X[i, :]-mu)
        cov[k, :] = (1/(N-k-1))*cov[k, :]
        
    return cov

def autocorrelation(autocov): # Autocorrelation of a Markov chain, calculated from its autocovariance
    corr = np.zeros(np.shape(autocov))
    for i in range(int(len(autocov[0, :, 0]))):
        corr[:, i, 0] = np.divide(autocov[:, i, 0], np.maximum(np.abs(autocov[:, 0, 0]), np.finfo(np.float64).eps))
        corr[:, i, 1] = np.divide(autocov[:, i, 1], np.maximum(np.abs(autocov[:, 0, 1]), np.finfo(np.float64).eps))
    
    return corr

def get_M(autocov): # Last element of the chain used to calculate sigma^2_mcmc
    M1 = None
    for k in range(int(len(autocov[:, 0])/2)):
        if ((autocov[2*k, 0] + autocov[2*k+1, 0])<=0):
            M1 = 2*k
            break
    if M1 is None:
        M1 = int(len(autocov[:, 0])-2)
    M2 = None
    for j in range(int(len(autocov[:, 1])/2)):
        if ((autocov[2*j, 1] + autocov[2*j+1, 1])<=0) :
            M2 = 2*j
            break
    if M2 is None:
        M2 = int(len(autocov[:, 1])-2)
    
    return M1, M2

def get_sigma(autocov): # Time-average variance constant of a Markov chain, calculated from its autocovariance
    [M1, M2] = get_M(autocov)
    id1 = range(1, M1)
    S1 = autocov[0, 0] + 2*np.sum(autocov[id1, 0])
    id2 = range(1, M2)
    S2 = autocov[0, 1] + 2*np.sum(autocov[id2, 1])

    return [S1, S2]

def effective_sample_size(autocov): # effective sample size of a Markov chain, calculated from its autocovariance
    [n, N] = np.shape(autocov[:, :, 0])
    ess = np.zeros([n, 2])
    for i in range(n):
        sigma = get_sigma(autocov[i, :, :])
        ess[i, 0] = N*np.divide(autocov[i, 0, 0], np.maximum(sigma[0],  np.finfo(np.float64).eps))
        ess[i, 1] = N*np.divide(autocov[i, 0, 1], np.maximum(sigma[1],  np.finfo(np.float64).eps))

    return ess