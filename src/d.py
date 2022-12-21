import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def f(q1, q2, alpha):
    return np.exp(-alpha*(q1**2+q2**2-1/4)**2)

def U(q, alpha):
    return alpha*(q[0]**2+q[1]**2-1/4)**2

def K(p, m):
    return np.sum(1/2*np.divide(p*p, m))

def dU(q, alpha):
    dU1 = 4*q[0]*alpha*(q[0]**2+q[1]**2-1/4)
    dU2 = 4*q[1]*alpha*(q[0]**2+q[1]**2-1/4)
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



def Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha):
    q = np.zeros([N+1, 2])
    q[0, :]=q0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=m)
        q_star, p_star = Verlet(q[i, :], p, T, eps, alpha, m)
        u = np.random.uniform()
        if (u<np.exp(-U(q_star, alpha)+U(q[i, :], alpha)-K(p_star, m)+K(p, m))):
            q[i+1, :] = q_star
        else:
            q[i+1, :] = q[i, :]
    
    return q

    

    

def main():
    q0 = [0, 0]
    eps = 0.01
    alpha = 10**1
    m = [1, 1]
    T = 1
    N = 1000

    n = 1
    final_q = np.zeros([n, 2])

    #final_q[:, :] = Hamiltonian_Monte_Carlo(q0, m, N, eps, alpha)[-1, :]
    #print(np.shape(final_q))
    
    for i in range(n):
        q = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
        final_q[i, :] = q[-1, :]
        if (i+1)%100 == 0:
            print("Iteration: ", i+1, "/", n)
    
    #q = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)

    t = np.linspace(0, T, N+1)
    v = np.exp(-alpha*(q[:, 0]**2+q[:, 1]**2-1/4)**2)
    x = np.linspace(-1, 1, 100)
    y = np.zeros([100, 100])
    for i in range(100):
        y[:, i] = np.exp(-alpha*(x[:]**2+x[i]**2-1/4)**2)

    fig, axs = plt.subplots(2, 2) 
    axs[0, 0].hist(final_q[:, 0], 20, density=False)
    axs[0, 1].hist(final_q[:, 1], 20, density=False)
    axs[1, 0].hist2d(final_q[:, 0], final_q[:, 1], bins=(20, 20), cmap=plt.cm.jet)
    axs[1, 1].pcolormesh(x, x, y, cmap=plt.cm.jet)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, q[:, 0])
    ax2.plot(t, q[:, 1])

    plt.show()

    




if __name__=='__main__':
    main()