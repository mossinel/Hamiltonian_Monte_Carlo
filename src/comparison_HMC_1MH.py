from e import*



def main():
    
    X, y = dataset_import()
    ## Parameters
    sigma = 1000
    D = X.shape[1]

    q0 = np.zeros((1, D))

    eps = 0.01
    m = np.ones(D)*1
    T = 0.4
    
    N = 2000 #length of the chain
    var = 0.5
    B = 100

    sigma = 1000

    
    numb = 50 # number of chains simulated

    big_q = np.zeros([numb, N+1, D]) # matrix with all chains for 1MH
    big_qh = np.zeros([numb, N+1, D]) # matrix with all chains for HMC

    for i in range(numb):
        q, _ = MH_one_at_a_time(q0, N, var, X, y, sigma)
        qh, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)

        big_q[i, :, :] = q
        big_qh[i, :, :] = qh

        if (i+1)%(int(np.floor(numb/10))) == 0:
            print("Iteration: ", i+1, "/", numb)


    exp_q = np.mean(big_q, axis=0) # mean over all chains of q(t) for 1MH

    exp_qh = np.mean(big_qh, axis=0) # mean over all chains of q(t) for HMC

    vn = range(N+1)

  

    v = [0, 1, 2, 5]

    fig, ax0 = plt.subplots(1, 1)
    ax0.plot(vn, exp_q[:, v], label=("MH intercept", "MH age", "MH lwt", "MH smoke"))
    ax0.plot(vn, exp_qh[:, v], label=("HMC intercept", "HMC age", "HMC lwt", "HMC smoke"))
    ax0.set_title("Average q")
    ax0.set_ylabel("q")
    ax0.set_xlabel("N of the chain")
    ax0.legend()

    
       

    plt.show()







if __name__=="__main__":
    main()