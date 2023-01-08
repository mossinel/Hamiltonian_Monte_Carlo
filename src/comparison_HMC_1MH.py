from e_new_new import*



def main():
    
    X, y = dataset_import()
    sigma = 1000
    D = X.shape[1]
    q0 = np.zeros((1, D))

    eps = 0.01
    m = np.ones(D)*1
    T = 0.4
    NH = 200
    BH = 100
    
    N = 2000
    var = 0.5
    B = 100

    sigma = 1000

    
    numb = 50

    big_q = np.zeros([numb, N+1, D])
    big_qh = np.zeros([numb, N+1, D])

    for i in range(numb):
        q, _ = MH_one_at_a_time(q0, N, var, X, y, sigma)
        qh, _ = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)

        big_q[i, :, :] = q
        big_qh[i, :, :] = qh

        if (i+1)%(int(np.floor(numb/10))) == 0:
            print("Iteration: ", i+1, "/", numb)


    exp_q = np.mean(big_q, axis=0)
    var_q = np.var(big_q, axis=0)

    exp_qh = np.mean(big_qh, axis=0)
    var_qh = np.var(big_qh, axis=0)

    idx = B+np.asarray(range(N-B))
    #print(idx)
    tail_q = big_q[:, idx, :]
    tail_qh = big_qh[:, idx, :]
    var_tail_q = np.var(tail_q, axis=0)
    print("Calculating covariance...")
    cov = np.zeros([numb, len(idx)-1, D])
    cov_h = np.zeros([numb, len(idx)-1, D])
    for i in range(numb):
        cov[i, :, :] = autocovariance(tail_q[i, :, :])
        cov_h[i, :, :] = autocovariance(tail_qh[i, :, :])
        if (i+1)%(int(np.floor(numb/10))) == 0:
            print("Iteration: ", i+1, "/", numb)
    corr = autocorrelation(cov)
    corr_h = autocorrelation(cov_h)

    vn = range(N+1)

  

    v = [0, 1, 2, 5]

    fig, ax0 = plt.subplots(1, 1)
    ax0.plot(vn, exp_q[:, v], label=("MH intercept", "MH age", "MH lwt", "MH smoke"))
    ax0.plot(vn, exp_qh[:, v], label=("HMC intercept", "HMC age", "HMC lwt", "HMC smoke"))
    ax0.set_title("Average q")
    ax0.set_ylabel("q")
    ax0.set_xlabel("N of the chain")
    ax0.legend()

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(np.mean(corr[:, :, 0:4], axis=0), label=("RWMH q(0)", "RWMH q(1)", "RWMH q(2)", "RWMH q(3)"))
    ax1.set_title("Covariance for q, averaged")


    ax1.plot(np.mean(corr_h[:, :, 0:4], axis=0), label=("HMC q(0)", "HMC q(1)", "HMC q(2)", "HMC q(3)"))
    ax1.legend()
    
       

    plt.show()







if __name__=="__main__":
    main()