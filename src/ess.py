from d import*

def main():
    alpha = 10**1
    q0 = [0, 0]

    number = 51 


    vector_N = np.ndarray.astype(np.floor(np.linspace(50, 500, number)), dtype=int)
    vector_N2 = vector_N

    n = 20 
    
    eps = 0.01
    m = [0.1, 0.1]
    T = 0.2
    

    sigma = 0.1


    ESS = np.zeros([number, 2])
    ESS_ham = np.zeros([number, 2])

    k=0

    for N in vector_N:
        B = 20
        big_q = np.zeros([n, vector_N2[k]+1, 2])
        big_q_ham = np.zeros([n, N+1, 2])
        for i in range(n):
            q_ham, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
            q, _ = Metropolis_Hastings(q0, vector_N2[k], alpha, sigma)
            
            big_q[i, :, :] = q
            big_q_ham[i, :, :] = q_ham
        
        idx = B+np.asarray(range(N-B))
        idx_2 = B+np.asarray(range(vector_N2[k]-B))
        tail_q = big_q[:, idx_2, :]
        tail_q_ham = big_q_ham[:, idx, :]


        cov = np.zeros([n, len(idx_2)-1, 2])
        cov_ham = np.zeros([n, len(idx)-1, 2])
        for i in range(n):
            cov[i, :, :] = autocovariance(tail_q[i, :, :])
            cov_ham[i, :, :] = autocovariance(tail_q_ham[i, :, :])


        ESS[k, :] = np.mean(effective_sample_size(cov), axis=0)
        
        ESS_ham[k, :] = np.mean(effective_sample_size(cov_ham), axis=0)

        print("Iteration: ", k+1, "/", len(vector_N))
        k=k+1

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(2*vector_N2, ESS[:, 0])
    ax[1].plot((2+np.floor(T/eps)*4*vector_N), ESS_ham[:, 0])
    ax[0].set_title("ESS q[0], RWMH")
    ax[1].set_title("ESS q[0], HMC")
    ax[0].set_ylabel("Effective sample size")
    ax[1].set_ylabel("Effective sample size")
    fig.tight_layout()



    plt.show()



if __name__=="__main__":
    main()