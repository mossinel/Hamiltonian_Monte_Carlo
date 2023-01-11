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
    # Calculate the
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


def Metropolis_Hastings(q0, N, alpha, sigma): # implementation of the Random Walk Metropolis-Hastings algorithm
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



def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha): # implementation of the Hamiltonian Monte Carlo algorithm
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

   

def main(): # used to test the first implementations, not used to produce the graphs in the report
    q0 = [0.0, 0.0]
    eps = 0.01
    alpha = 10**3
    m = [1, 1]
    T = 0.1
    N = 100
    sigma = 0.4
    
    
    n = 1000
    final_q = np.zeros([n, 2])
    final_q_ham = np.zeros([n, 2])
    ratio = np.zeros(n)
    big_q = np.zeros([n, N+1, 2])
    big_q_ham = np.zeros([n, N+1, 2])

        
    for i in range(n):

        q_ham, ratio[i] = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
        q, _ = Metropolis_Hastings(q0, N, alpha, sigma)
        final_q[i, :] = q[-1, :]
        final_q_ham[i, :] = q_ham[-1, :]
        big_q[i, :, :] = q
        big_q_ham[i, :, :] = q_ham
        if (i+1)%(int(np.floor(n/10))) == 0:
            print("Iteration: ", i+1, "/", n)

            
    
    exp_q = np.mean(big_q, axis=0)
    var_q = np.var(big_q, axis=0)

    # computing the unnormalized distribution

    t = np.linspace(0, T*N, N+1)
    v = np.exp(-alpha*(q[:, 0]**2+q[:, 1]**2-1/4)**2)
    x = np.linspace(-1, 1, 50)
    y = np.zeros([50, 50])
    for i in range(50):
        y[:, i] = np.exp(-alpha*(x[:]**2+x[i]**2-1/4)**2)


    ## Calculate autocorrelation of both chains q(:, 0) and q(:, 1)
    B = 20
    idx = B+np.asarray(range(N-B))

    tail_qx = big_q[:, idx, 0]
    tail_qy = big_q[:, idx, 1]
    tail_q = big_q[:, idx, :]
    tail_qx_ham = big_q_ham[:, idx, 0]
    tail_qy_ham = big_q_ham[:, idx, 1]

    print("Calculating covariance...")
    cov = np.zeros([n, len(idx)-1, 2])
    for i in range(n):
        cov[i, :, :] = autocovariance(tail_q[i, :, :])
    print("Calculating correlation...")
    corr = np.mean(autocorrelation(cov), axis=0)


    ## Calculate effective sample size
    print("Calculating ESS...")
    ESS = effective_sample_size(cov)
    ESS = np.sort(ESS, axis=0)
    id_min_ESS = int(np.floor(n/40))
    id_ESS = id_min_ESS+np.asarray(range(n-2*id_min_ESS))
    ESS_reduced = ESS[id_ESS, :]
    

    avx = np.reshape(tail_qx, -1)
    avy = np.reshape(tail_qy, -1)
    avx_ham = np.reshape(tail_qx_ham, -1)
    avy_ham = np.reshape(tail_qy_ham, -1)

    fig, axs = plt.subplots(2, 2) 
    axs[0, 0].hist2d(avx, avy, bins=(50, 50), cmap=plt.cm.jet)
    axs[0, 1].hist2d(avx_ham, avy_ham, bins=(50, 50), cmap=plt.cm.jet)
    axs[1, 0].hist2d(final_q[:, 0], final_q[:, 1], bins=(50, 50), cmap=plt.cm.jet)
    axs[1, 1].pcolormesh(x, x, y, cmap=plt.cm.jet)

    
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(t, q[:, 0])
    ax2.plot(t, q[:, 1])
    ax3.hist(ratio, 50, density=False)

    fig, axs2 = plt.subplots(2, 4)
    axs2[0, 0].plot(exp_q[:, 0])
    axs2[1, 0].plot(exp_q[:, 1])
    axs2[0, 1].plot(var_q[:, 0])
    axs2[1, 1].plot(var_q[:, 1])
    axs2[0, 2].plot(cov[-1, :, 0])
    axs2[1, 2].plot(cov[-1, :, 1])
    axs2[0, 3].plot(corr[:, 0])
    axs2[1, 3].plot(corr[:, 1])


    fig, ax3 = plt.subplots(1, 2)
    ax3[0].hist(ESS_reduced[:, 0], 50, density=False)
    ax3[1].hist(ESS_reduced[:, 1], 50, density=False)


    plt.show()


def test_f():
    n=200
    x = np.linspace(-2, 2, n)
    alpha = 10
    x_vec = np.asarray([x, x])
    print(np.shape(x_vec))
    y = np.zeros(n)
    for i in range(n):
        y[i] = f(x_vec[:, i], alpha)

    plt.plot(x, y)
    plt.show()


if __name__=='__main__':
    main()
