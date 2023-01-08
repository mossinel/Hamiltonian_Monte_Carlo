from d import*




def main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False):
    


    
    final_q = np.zeros([n, 2])
    final_q_ham = np.zeros([n, 2])
    ratio = np.zeros(n)
    ratio_ham = np.zeros(n)
    big_q = np.zeros([n, N+1, 2])
    big_q_ham = np.zeros([n, N+1, 2])

   
    
    for i in range(n):
        
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
    tail_qx = big_q[:, idx, 0]
    tail_qy = big_q[:, idx, 1]
    tail_q = big_q[:, idx, :]
    tail_qx_ham = big_q_ham[:, idx, 0]
    tail_qy_ham = big_q_ham[:, idx, 1]
    tail_q_ham = big_q_ham[:, idx, :]
    

    avx = np.reshape(tail_qx, -1)
    avy = np.reshape(tail_qy, -1)
    avx_ham = np.reshape(tail_qx_ham, -1)
    avy_ham = np.reshape(tail_qy_ham, -1)

    


    if "Density_end" in plot:
        fig1, ax1 =plt.subplots(1, 2, figsize=(8, 4))
        ax1[0].hist2d(big_q[:, -1, 0], big_q[:, -1, 1], range=[[-1, 1], [-1, 1]], bins=(50, 50), cmap=plt.cm.jet)
        ax1[1].hist2d(big_q_ham[:, -1, 0], big_q_ham[:, -1, 1], range=[[-1, 1], [-1, 1]], bins=(50, 50), cmap=plt.cm.jet)
        ax1[0].set_title("RWMH")
        ax1[1].set_title("HMC")
        ax1[0].set_xlabel("q[0]")
        ax1[1].set_xlabel("q[0]")
        ax1[0].set_ylabel("q[1]")
        ax1[1].set_ylabel("q[1]")
        fig1.tight_layout()
        

    if "Density_all" in plot:
        fig2, ax2 = plt.subplots(1, 3, figsize=(12,4))
        ax2[0].hist2d(avx, avy, range=[[-1, 1], [-1, 1]], bins=(50, 50), cmap=plt.cm.jet)
        ax2[1].hist2d(avx_ham, avy_ham, range=[[-1, 1], [-1, 1]], bins=(50, 50), cmap=plt.cm.jet)
        ax2[2].pcolormesh(x, x, y, cmap=plt.cm.jet)
        ax2[0].set_title("RWMH")
        ax2[1].set_title("HMC")
        ax2[2].set_title("Theoretical")
        ax2[0].set_xlabel("q[0]")
        ax2[1].set_xlabel("q[0]")
        ax2[2].set_xlabel("q[0]")
        ax2[0].set_ylabel("q[1]")
        ax2[1].set_ylabel("q[1]")
        ax2[2].set_ylabel("q[1]")
        fig2.tight_layout()

    if "RWMH" in plot:
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        ax.hist2d(avx, avy, range=[[-1, 1], [-1, 1]], bins=(50, 50), cmap=plt.cm.jet)
        ax.set_title("RWMH")
        ax.set_xlabel("q[0]")
        ax.set_ylabel("q[1]")
        fig.tight_layout()

    if "HMC" in plot:
        figh, axh = plt.subplots(1, 1, figsize=(4,4))
        axh.hist2d(avx_ham, avy_ham, range=[[-1, 1], [-1, 1]], bins=(50, 50), cmap=plt.cm.jet)
        axh.set_title("HMC")
        axh.set_xlabel("q[0]")
        axh.set_ylabel("q[1]")
        figh.tight_layout()


    print("Ratio RWMH: ", np.mean(ratio))
    print("Ratio HMX: ", np.mean(ratio_ham))


    

    plt.show()




if __name__=="__main__":

    alpha = 10**3
    q0 = [0, 0]
    q0_type = "Dirac" # Dirac or Normal
    offset = False #Offset of q0 [0.5, 0.5]

    N = 200 #length of the chain
    n = 1000 #number of chains simulated
    
    eps = 0.01
    m = [1, 1]
    T = 0.1
    
    sigma = 0.1
    
    B = 80
    
    plot=["Density_all", "Density_end", "HMC"]
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)
    
    m = [5, 5]
    T = 0.1
    plot="HMC"
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)

    m = [10, 1]
    T = 0.1
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)

    m = [5, 5]
    T=0.5
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)

    eps=0.05
    T = 0.5
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)

    T=0.1
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)
    
    T=0.1
    eps=0.01
    m = [0.1, 0.1]
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)
    
    plot=["RWMH"]
    sigma=0.4
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)
    
    sigma=0.02
    main(alpha, q0, N, n, eps, m, T, sigma, B, plot, q0_type="Dirac", offset=False)











