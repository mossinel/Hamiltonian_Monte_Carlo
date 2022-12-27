from d import*

def main():
    alpha = 10**1
    q0 = [0, 0]
    q0_type = "Dirac" # Dirac or Normal
    offset = False #Offset of q0 [0.5, 0.5]

    N = 80 #length of the chain

    number = 11
    vector_n = np.ndarray.astype(np.floor(np.linspace(200, 1000, number)), dtype=int) #number of chains simulated
    #vector_n = vector_n.astype(int)
    eps = 0.01
    m = [1, 1]
    T = 0.1
    
    sigma = 0.1
    

    ESS = np.zeros(number)
    ESS_ham = np.zeros(number)

    k=0

    for n in vector_n:
        B = 20
        big_q = np.zeros([n, N+1, 2])
        big_q_ham = np.zeros([n, N+1, 2])
        for i in range(n):
            if (q0_type=="Normal"):
                q0 = np.random.normal(size=2)/4
            if offset:
                q0 = q0+[0.5, 0.5]
            q_ham, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
            q, _ = Metropolis_Hastings(q0, N, alpha, sigma)
            
            big_q[i, :, :] = q
            big_q_ham[i, :, :] = q_ham
        
        idx = B+np.asarray(range(N-B))
        #print(idx)
        tail_q = big_q[:, idx, :]
        tail_q_ham = big_q_ham[:, idx, :]


        cov = np.zeros(np.shape(tail_q))
        cov_ham = np.zeros(np.shape(tail_q_ham))
        for i in range(n):
            cov[i, :, :] = autocovariance(tail_q[i, :, :])
            cov_ham[i, :, :] = autocovariance(tail_q_ham[i, :, :])


        ESS[k] = np.mean(effective_sample_size(cov))
        
        ESS_ham[k] = np.mean(effective_sample_size(cov_ham))

        print("Iteration: ", k+1, "/", len(vector_n))
        k=k+1

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(2*N*2*vector_n, ESS)
    ax[1].plot(N*(2+np.floor(T/eps)*4*vector_n), ESS_ham)
    ax[0].set_title("ESS, RWMH")
    ax[1].set_title("ESS, HMC")

    plt.show()



if __name__=="__main__":
    main()