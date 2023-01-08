from e_new_new import*

def main():
    
    X, y = dataset_import()

    D = X.shape[1]
    q0 = np.zeros((1, D))
    eps = 0.01
    m = np.ones(D)*1
    T = 0.4

    
    sigma = 1000
    B = 200

    number = 11 # 11


    #N = 80 #length of the chain
    vector_N = np.ndarray.astype(np.floor(np.linspace(400, 6000, number)), dtype=int)

    n = 10 


    
    ESS_ham = np.zeros([number, D])

    k=0

    for N in vector_N:
        big_q_ham = np.zeros([n, N+1, D])
        for i in range(n):
            
            q_ham, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)
            
            big_q_ham[i, :, :] = q_ham
        
        idx = B+np.asarray(range(N-B))
        #print(idx)
        tail_q_ham = big_q_ham[:, idx, :]


        #print(np.shape(cov))
        cov_ham = np.zeros([n, len(idx)-1, D])
        for i in range(n):
            cov_ham[i, :, :] = autocovariance(tail_q_ham[i, :, :])


        
        ESS_ham[k, :] = np.mean(effective_sample_size(cov_ham), axis=0)

        print("Iteration: ", k+1, "/", len(vector_N))
        k=k+1

    fig, ax = plt.subplots(1, 1)
    ax.plot(vector_N, ESS_ham[:, 0], label="intercept")
    ax.plot(vector_N, ESS_ham[:, 1], label="age")
    ax.plot(vector_N, ESS_ham[:, 2], label="lwt")
    ax.plot(vector_N, ESS_ham[:, 3], label="race2black")
    ax.plot(vector_N, ESS_ham[:, 4], label="race2other")
    ax.plot(vector_N, ESS_ham[:, 5], label="smoke")
    ax.plot(vector_N, ESS_ham[:, 6], label="ptl")
    ax.plot(vector_N, ESS_ham[:, 7], label="ht")
    ax.plot(vector_N, ESS_ham[:, 8], label="ui")
    ax.plot(vector_N, ESS_ham[:, 9], label="ftv21")
    ax.plot(vector_N, ESS_ham[:, 10], label="ftv22+")
    ax.set_title("ESS, HMC")
    ax.set_ylabel("Effective sample size")
    ax.set_xlabel("N of the chain")
    ax.legend()

    plt.show()



if __name__=="__main__":
    main()