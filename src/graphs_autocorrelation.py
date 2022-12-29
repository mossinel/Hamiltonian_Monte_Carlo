from d import*




def main():

    alpha = 10**1
    q0 = [0, 0]
    q0_type = "Dirac" # Dirac or Normal
    offset = False #Offset of q0 [0.5, 0.5]

    N = 100 #length of the chain
    n = 1000 #number of chains simulated
    
    eps = 0.01
    m = [1, 1]
    T = 0.1
    
    sigma = 0.1
    
    B = 20
    
    final_q = np.zeros([n, 2])
    final_q_ham = np.zeros([n, 2])
    ratio = np.zeros(n)
    ratio_ham = np.zeros(n)
    big_q = np.zeros([n, N+1, 2])
    big_q_ham = np.zeros([n, N+1, 2])

    #final_q[:, :] = Hamiltonian_Monte_Carlo(q0, m, N, eps, alpha)[-1, :]
    #print(np.shape(final_q))

    
    for i in range(n):
        if (q0_type=="Normal"):
            q0 = np.random.normal(size=2)/4
        if offset:
            q0 = q0+[0.5, 0.5]
        q_ham, ratio_ham[i] = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, alpha)
        q, ratio[i] = Metropolis_Hastings(q0, N, alpha, sigma)
        final_q[i, :] = q[-1, :]
        final_q_ham[i, :] = q_ham[-1, :]
        big_q[i, :, :] = q
        big_q_ham[i, :, :] = q_ham
        if (i+1)%(int(np.floor(n/10))) == 0:
            print("Iteration: ", i+1, "/", n)



    exp_q = np.mean(big_q, axis=0)
    var_q = np.var(big_q, axis=0)

    exp_q_ham = np.mean(big_q_ham, axis=0)
    var_q_ham = np.var(big_q_ham, axis=0)
    # Preparing for the graph of f

    t = np.linspace(0, T*N, N+1)
    v = np.exp(-alpha*(q[:, 0]**2+q[:, 1]**2-1/4)**2)
    x = np.linspace(-1, 1, 50)
    y = np.zeros([50, 50])
    for i in range(50):
        y[:, i] = np.exp(-alpha*(x[:]**2+x[i]**2-1/4)**2)
    
    
    idx = B+np.asarray(range(N-B))
    #print(idx)
    tail_qx = big_q[:, idx, 0]
    tail_qy = big_q[:, idx, 1]
    tail_q = big_q[:, idx, :]
    tail_qx_ham = big_q_ham[:, idx, 0]
    tail_qy_ham = big_q_ham[:, idx, 1]
    tail_q_ham = big_q_ham[:, idx, :]
    var_tail_q = np.var(tail_q, axis=0)
    

    avx = np.reshape(tail_qx, -1)
    avy = np.reshape(tail_qy, -1)
    avx_ham = np.reshape(tail_qx_ham, -1)
    avy_ham = np.reshape(tail_qy_ham, -1)

    print("Calculating covariance...")
    cov = np.zeros([n, len(idx)-1, 2])
    cov_ham = np.zeros([n, len(idx)-1, 2])
    for i in range(n):
        cov[i, :, :] = autocovariance(tail_q[i, :, :])
        cov_ham[i, :, :] = autocovariance(tail_q_ham[i, :, :])
        if (i+1)%(int(np.floor(n/10))) == 0:
            print("Iteration: ", i+1, "/", n)
    corr = autocorrelation(cov)
    corr_ham = autocorrelation(cov_ham)
    
    
    


    ## Calculate effective sample size
    print("Calculating ESS...")
    ESS = effective_sample_size(cov)
    #ESS = np.sort(ESS, axis=0)
    
    ESS_ham = effective_sample_size(cov_ham)
    #ESS_ham = np.sort(ESS_ham, axis=0)

    id_min_ESS = int(np.floor(n/40))
    id_ESS = id_min_ESS+np.asarray(range(n-2*id_min_ESS))
    ESS_reduced = ESS[id_ESS, :]
    ESS_reduced_ham = ESS_ham[id_ESS, :]
    
    fig, ax0 = plt.subplots(2, 2)
    ax0[0, 0].plot(exp_q[:, 0], label="RWMH")
    ax0[0, 1].plot(exp_q[:, 1], label="RWMH")
    ax0[1, 0].plot(var_q[:, 0], label="RWMH")
    ax0[1, 1].plot(var_q[:, 1], label="RWMH")
    ax0[0, 0].set_title("Average q[0](t)")
    ax0[0, 1].set_title("Average q[1](t)")
    ax0[1, 0].set_title("Variance q[0](t)")
    ax0[1, 1].set_title("Variance q[1](t)")

    #fig, ax0_ham = plt.subplots(2, 2)
    ax0[0, 0].plot(exp_q_ham[:, 0], label="HMC")
    ax0[0, 1].plot(exp_q_ham[:, 1], label="HMC")
    ax0[1, 0].plot(var_q_ham[:, 0], label="HMC")
    ax0[1, 1].plot(var_q_ham[:, 1], label="HMC")
    ax0[0, 0].legend()
    ax0[0, 1].legend()
    ax0[1, 0].legend()
    ax0[1, 1].legend()
    #ax0_ham[0, 0].set_title("Average q[0](t), HMC")
    #ax0_ham[0, 1].set_title("Average q[1](t), HMC")
    #ax0_ham[1, 0].set_title("Variance q[0](t), HMC")
    #ax0_ham[1, 1].set_title("Variance q[1](t), HMC")

    fig, ax1 = plt.subplots(2, 2)
    ax1[0, 0].plot(cov[-1, :, 0], label="RWMH")
    ax1[0, 1].plot(cov[-1, :, 1], label="RWMH")
    ax1[1, 0].plot(corr[-1, :, 0], label="RWMH")
    ax1[1, 1].plot(corr[-1, :, 1], label="RWMH")
    ax1[0, 0].set_title("Covariance for q[0], 1 chain")
    ax1[0, 1].set_title("Covariance for q[1], 1 chain")
    ax1[1, 0].set_title("Correlation for q[0], 1 chain")
    ax1[1, 1].set_title("Correlation for q[1], 1 chain")

    #fig, ax1_ham = plt.subplots(2, 2)
    ax1[0, 0].plot(cov_ham[-1, :, 0], label="HMC")
    ax1[0, 1].plot(cov_ham[-1, :, 1], label="HMC")
    ax1[1, 0].plot(corr_ham[-1, :, 0], label="HMC")
    ax1[1, 1].plot(corr_ham[-1, :, 1], label="HMC")
    ax1[0, 0].legend()
    ax1[0, 1].legend()
    ax1[1, 0].legend()
    ax1[1, 1].legend()
    #ax1_ham[0, 0].set_title("Covariance for q[0], 1 chain, HMC")
    #ax1_ham[0, 1].set_title("Covariance for q[1], 1 chain, HMC")
    #ax1_ham[1, 0].set_title("Correlation for q[0], 1 chain, HMC")
    #ax1_ham[1, 1].set_title("Correlation for q[1], 1 chain, HMC")
    


    fig, ax2 = plt.subplots(2, 2)
    ax2[0, 0].plot(np.mean(cov[:, :, 0], axis=0), label="RWMH")
    ax2[0, 1].plot(np.mean(cov[:, :, 1], axis=0), label="RWMH")
    ax2[1, 0].plot(np.mean(corr[:, :, 0], axis=0), label="RWMH")
    ax2[1, 1].plot(np.mean(corr[:, :, 1], axis=0), label="RWMH")
    ax2[0, 0].set_title("Averaged covariance for q[0]")
    ax2[0, 1].set_title("Averaged covariance for q[1]")
    ax2[1, 0].set_title("Averaged correlation for q[0]")
    ax2[1, 1].set_title("Averaged correlation for q[1]")

    #fig, ax2_ham = plt.subplots(2, 2)
    ax2[0, 0].plot(np.mean(cov_ham[:, :, 0], axis=0), label="HMC")
    ax2[0, 1].plot(np.mean(cov_ham[:, :, 1], axis=0), label="HMC")
    ax2[1, 0].plot(np.mean(corr_ham[:, :, 0], axis=0), label="HMC")
    ax2[1, 1].plot(np.mean(corr_ham[:, :, 1], axis=0), label="HMC")
    ax2[0, 0].legend()
    ax2[0, 1].legend()
    ax2[1, 0].legend()
    ax2[1, 1].legend()
    #ax2_ham[0, 0].set_title("Averaged covariance for q[0], HMC")
    #ax2_ham[0, 1].set_title("Averaged covariance for q[1], HMC")
    #ax2_ham[1, 0].set_title("Averaged correlation for q[0], HMC")
    #ax2_ham[1, 1].set_title("Averaged correlation for q[1], HMC")



    fig, ax3 = plt.subplots(2, 2)
    ax3[0, 0].hist(ESS[:, 0], 50, density=False)
    ax3[0, 1].hist(ESS[:, 1], 50, density=False)
    ax3[1, 0].hist(ESS_ham[:, 0], 50, density=False)
    ax3[1, 1].hist(ESS_ham[:, 1], 50, density=False)
    ax3[0, 0].set_title("ESS for q[0], RWMH")
    ax3[0, 1].set_title("ESS for q[1], RWMH")
    ax3[1, 0].set_title("ESS for q[0], HMC")
    ax3[1, 1].set_title("ESS for q[1], HMC")
    

    plt.show()




if __name__=="__main__":
    main()