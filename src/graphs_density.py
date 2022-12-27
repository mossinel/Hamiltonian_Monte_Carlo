from d import*




def main():

    alpha = 10**1
    q0 = [0, 0]
    q0_type = "Dirac" # Dirac or Normal
    offset = False #Offset of q0 [0.5, 0.5]

    N = 100 #length of the chain
    n = 2000 #number of chains simulated
    
    eps = 0.05
    m = [5, 5]
    T = 0.5
    
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
    #var_tail_q = np.var(tail_q, axis=0)
    

    avx = np.reshape(tail_qx, -1)
    avy = np.reshape(tail_qy, -1)
    avx_ham = np.reshape(tail_qx_ham, -1)
    avy_ham = np.reshape(tail_qy_ham, -1)

    

    

    fig, ax1 =plt.subplots(1, 2)
    ax1[0].hist2d(big_q[:, -1, 0], big_q[:, -1, 1], bins=(50, 50), cmap=plt.cm.jet)
    ax1[1].hist2d(big_q_ham[:, -1, 0], big_q_ham[:, -1, 1], bins=(50, 50), cmap=plt.cm.jet)
    ax1[0].set_title("RWMC")
    ax1[1].set_title("HMC")


    fig, ax2 = plt.subplots(1, 3)
    ax2[0].hist2d(avx, avy, bins=(50, 50), cmap=plt.cm.jet)
    ax2[1].hist2d(avx_ham, avy_ham, bins=(50, 50), cmap=plt.cm.jet)
    ax2[2].pcolormesh(x, x, y, cmap=plt.cm.jet)
    ax2[0].set_title("RWMC")
    ax2[1].set_title("HMC")
    ax2[2].set_title("Theoretical")

    print("Ratio RWMH: ", np.mean(ratio))
    print("Ratio HMX: ", np.mean(ratio_ham))

    

    plt.show()




if __name__=="__main__":
    main()