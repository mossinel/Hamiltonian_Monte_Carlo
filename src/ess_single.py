from d import*

def main():
    alpha = 10**3
    q0 = [0, 0]
    q0_type = "Dirac" # Dirac or Normal
    offset = False #Offset of q0 [0.5, 0.5]

    number = 11 # 16

    #N = 80 #length of the chain
    vector_N = np.ndarray.astype(np.floor(np.linspace(70, 470, number)), dtype=int)
    vector_N2 = vector_N
    #vector_N =np.asarray([50, 50, 50])
    #print(vector_N)

    n = 50 #200
    #vector_n = np.ndarray.astype(np.floor(np.linspace(200, 1000, number)), dtype=int) #number of chains simulated
    
    eps = 0.01
    m = [0.1, 0.1]
    T = 0.2
    

    sigma = 0.1


    ESS = np.zeros([number, 2])
    ESS_ham = np.zeros([number, 2])

    k=0

    for N in vector_N:
        B = 50
        big_q = np.zeros([n, vector_N2[k]+1, 2])
        big_q_ham = np.zeros([n, N+1, 2])
        for i in range(n):
            if (q0_type=="Normal"):
                q0 = np.random.normal(size=2)/4
            if offset:
                q0 = q0+np.asarray([0.5, 0.5])
            q_ham, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
            q, _ = Metropolis_Hastings(q0, vector_N2[k], alpha, sigma)
            
            big_q[i, :, :] = q
            big_q_ham[i, :, :] = q_ham
        
        idx = B+np.asarray(range(N-B))
        idx_2 = B+np.asarray(range(vector_N2[k]-B))
        #print(idx)
        tail_q = big_q[:, idx_2, :]
        tail_q_ham = big_q_ham[:, idx, :]


        cov = np.zeros([n, len(idx_2)-1, 2])
        #print(np.shape(cov))
        cov_ham = np.zeros([n, len(idx)-1, 2])
        for i in range(n):
            cov[i, :, :] = autocovariance(tail_q[i, :, :])
            cov_ham[i, :, :] = autocovariance(tail_q_ham[i, :, :])


        ESS[k, :] = np.mean(effective_sample_size(cov), axis=0)
        
        ESS_ham[k, :] = np.mean(effective_sample_size(cov_ham), axis=0)

        print("Iteration: ", k+1, "/", len(vector_N))
        k=k+1

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(2*vector_N2, ESS[:, 0])
    ax[0, 1].plot(2*vector_N2, ESS[:, 1])
    ax[1, 0].plot((2+np.floor(T/eps)*4*vector_N), ESS_ham[:, 0])
    ax[1, 1].plot((2+np.floor(T/eps)*4*vector_N), ESS_ham[:, 1])
    ax[0, 0].set_title("ESS q[0], RWMH")
    ax[0, 1].set_title("ESS q[1], RWMH")
    ax[1, 0].set_title("ESS q[0], HMC")
    ax[1, 1].set_title("ESS q[1], HMC")
    ax[0, 0].set_ylabel("Effective sample size")
    ax[1, 0].set_xlabel("Nb of evaluation of f and df/dqi")
    ax[1, 0].set_ylabel("Effective sample size")
    ax[1, 1].set_xlabel("Nb of evaluation of f and df/dqi")

    plt.show()



if __name__=="__main__":
    main()