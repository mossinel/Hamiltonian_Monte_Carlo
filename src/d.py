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


def Verlet(q0, p0, eps, alpha, m):
    p_tmp = p0 - eps/2*dU(q0, alpha)
    q = q0+eps*np.divide(p_tmp, m)
    p = p_tmp - eps/2*dU(q, alpha)
    return q, p

def Metropolis_Hastings(f, alpha, eps, m1, m2, T):
    pass

def Hamiltonian_Monte_Carlo(q0, m, N, eps, alpha):
    q = np.zeros([N+1, 2])
    q[0]=q0
    for i in range(N):
        p = st.norm.rvs(loc=0, scale=m)
        q_star, p_star = Verlet(q[i], p, eps, alpha, m)
        u = np.random.uniform()
        if (u<np.exp(-U(q_star, alpha)+U(q[i], alpha)-K(p_star, m)+K(p, m))):
            q[i+1] = q_star
        else:
            q[i+1] = q[i]
    
    return q

    

    

def main():
    q0 = [0, 0]
    eps = 0.01
    alpha = 10**3
    m = [1, 1]
    T = 100
    N = int(np.floor(T/eps))

    q = Hamiltonian_Monte_Carlo(q0, m, N, eps, alpha)

    t = np.linspace(0, T, N+1)

    plt.plot(t, q[:, 1])

    plt.show()

    







if __name__=='__main__':
    main()