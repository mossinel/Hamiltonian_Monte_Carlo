import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(q, alpha):
    return np.exp(-alpha*(q@q-1/4)**2)

def U(q, alpha):
    return alpha*(q@q-1/4)**2

def K(p, m):
    return np.sum(1/2*np.divide(p*p, m))

def dU(q, alpha):
    dU1 = 4*q[0]*alpha*(q@q-1/4)
    dU2 = 4*q[1]*alpha*(q@q-1/4)
    return np.asarray([dU1, dU2])


def Verlet(q0, p0, eps, T, alpha, m):
    t=0
    q=q0
    p=p0
    while t<T:
        p_tmp = p - eps/2*dU(q, alpha)
        q = q + eps*np.divide(p_tmp, m)
        p = p_tmp - eps/2*dU(q, alpha)
        t=t+eps
    return q, p

def autocovariance(X):
    mu = np.mean(X, axis=0)
    N = len(X[:, 0])
    cov = np.zeros([N, 2])
    corr = 0
    for k in range(N):
        for i in range(N-k):
            cov[k, :] += ((X[i+k, :])-mu)*(X[i, :]-mu)
        cov[k, :] = (1/(N-1))*cov[k, :]
        
    return cov

def autocorrelation(autocov, var_q):
    mean_cov = np.mean(autocov, axis = 0)
    corr = np.divide(mean_cov, (var_q*var_q[-1]))
    return corr



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



def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha):
    q = np.zeros([N+1, 2])
    q[0, :] = q0
    accepted = 0
    rejected = 0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=m)
        q_star, p_star = Verlet(q[i, :], p, T, eps, alpha, m)
        u = np.random.uniform()
        if (u<np.exp(-U(q_star, alpha)+U(q[i, :], alpha)-K(p_star, m)+K(p, m))):
            q[i+1, :] = q_star
            accepted = accepted+1
        else:
            q[i+1, :] = q[i, :]
            rejected = rejected+1
            #print("Problem")
    ratio = accepted/(accepted+rejected)
    return q, ratio

def prepare_animation(bar_container):
    def animate(frame_number, q):
        data_x = q[frame_number, :, 0]
        data_y = q[frame_number, :, 1] 
        n, _ , _ , _ = np.histogram2d(data_x, data_y, bins=(50, 50), cmap=plt.cm.jet)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)

        return bar_container.patches
    
    return animate


        
  

def main():
    q0 = [0.0, 0.0] # idea: change q0, if q0 follow the objective distribution, then q should follow the distribution at any time
    eps = 0.01
    alpha = 10**2
    m = [1, 1]
    T = 0.10
    N = 200
    sigma = 0.1
    
    
    n = 1000
    final_q = np.zeros([n, 2])
    final_q_ham = np.zeros([n, 2])
    ratio = np.zeros(n)
    big_q = np.zeros([n, N+1, 2])

    #final_q[:, :] = Hamiltonian_Monte_Carlo(q0, m, N, eps, alpha)[-1, :]
    #print(np.shape(final_q))

    
    for i in range(n):
        #q0 = [0.5, 0.5]+np.random.normal(size=2)/2
        q_ham, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
        q, ratio[i] = Metropolis_Hastings(q0, N, alpha, sigma)
        final_q[i, :] = q[-1, :]
        final_q_ham[i, :] = q_ham[-1, :]
        big_q[i, :, :] = q
        if (i+1)%(int(np.floor(n/10))) == 0:
            print("Iteration: ", i+1, "/", n)

            
    
    #q = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
    exp_q = np.mean(big_q, axis=0)
    var_q = np.var(big_q, axis=0)

    t = np.linspace(0, T*N, N+1)
    v = np.exp(-alpha*(q[:, 0]**2+q[:, 1]**2-1/4)**2)
    x = np.linspace(-1, 1, 50)
    y = np.zeros([50, 50])
    for i in range(50):
        y[:, i] = np.exp(-alpha*(x[:]**2+x[i]**2-1/4)**2)


    ## Calculate autocorrelation of both chains

    flatt = 20
    idx = flatt+np.asarray(range(n-flatt))
    #print(idx)
    #tail_qx = big_q[idx, :, 0]
    #tail_qy = big_q[idx, :, 1]
    print("Calculating covariance...")
    cov = np.zeros(np.shape(big_q))
    for i in range(n):
        cov[i, :, :] = autocovariance(big_q[i, :, :])
    print("Calculating correlation...")
    corr = autocorrelation(cov, var_q)




    flatt = 20
    idx = flatt+np.asarray(range(n-flatt))
    #print(idx)
    #tail_qx = big_q[idx, :, 0]
    #tail_qy = big_q[idx, :, 1]
    #avx = np.reshape(tail_qx, -1)
    #avy = np.reshape(tail_qy, -1)

    fig, axs = plt.subplots(2, 2) 
    #axs[0, 0].hist(final_q[:, 0], 50, density=False)
    #axs[0, 1].hist(final_q[:, 1], 50, density=False)
    #axs[0, 0].hist2d(big_q[:, -3, 0], big_q[:, -3, 1], bins=(50, 50), cmap=plt.cm.jet)
    #axs[0, 1].hist2d(big_q[:, -2, 0], big_q[:, -2, 1], bins=(50, 50), cmap=plt.cm.jet)
    #axs[0, 0].hist2d(avx, avy, bins=(50, 50), cmap=plt.cm.jet)
    #axs[0, 1].hist2d(big_q[:, -1, 0], big_q[:, -1, 1], bins=(50, 50), cmap=plt.cm.jet)
    axs[0, 0].hist2d(final_q_ham[:, 0], final_q_ham[:, 1], bins=(50, 50), cmap=plt.cm.jet)
    axs[0, 1].hist2d(big_q[:, -2, 0], big_q[:, -2, 1], bins=(50, 50), cmap=plt.cm.jet)
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
    #axs2[0, 2].plot(np.mean(cov[-1, :, 0], axis=0))
    #axs2[1, 2].plot(np.mean(cov[-1, :, 1], axis=0))
    axs2[0, 2].plot(cov[-1, :, 0])
    axs2[1, 2].plot(cov[-1, :, 1])
    axs2[0, 3].plot(corr[:, 0])
    axs2[1, 3].plot(corr[:, 1])



    #f, ax = plt.subplots()
    #_, _, _,  bar_container = ax.hist2d(final_q[:, 0], final_q[:, 1], bins=(50, 50))
    #ax.set_ylim(top=100)
    #ani = animation.FuncAnimation(f, prepare_animation(bar_container, big_q), 50, repeat=False, blit=True)


    #idea: make a plot of the density over an axis as a function of the time


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
    #test_f()