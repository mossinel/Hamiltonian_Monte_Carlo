from e_new_new import*




def main():
    X, y = dataset_import()

    q0 = np.zeros((1, X.shape[1]))
    eps = 0.01
    m = np.ones(X.shape[1])
    T = 0.10
    sigma = 1000
    
    number = 30
    vector_N = np.ndarray.astype(np.floor(np.logspace(3, 5, number)),int)

    B = 500

    q_mean = np.zeros([number, 11])
    q_var = np.zeros([number, 11])

    k=0
    for N in vector_N:
        (q, _) = Hamiltonian_Monte_Carlo(q0, m, N, T, eps, X, y, sigma)
        
        #q_mean[k, :] = np.mean(q[B:, :], axis=0)
        q_var[k, :] = np.var(q[B:, :], axis=0)
        print("Iteration: ", k+1, "/", len(vector_N))
        k=k+1

    n=10
    q_ex = np.zeros([n, 11])
    
    #for i in range(n):
    #    q, _ = Hamiltonian_Monte_Carlo(q0, m, 50000, T, eps, X, y, sigma)
    #    q_ex[i, :] = np.mean(q[B:, :], axis=0)
    #    print("Iteration: ", i+1, "/", n)

    q_exact=np.mean(q_ex, axis=0)
    
    fig, ax = plt.subplots(1, 1)
    ax.loglog(vector_N, np.divide(np.mean(q_var, axis=1), np.sqrt(vector_N)))
    ax.set_xlabel("Length of the chain N")
    ax.set_ylabel("Variance/$\sqrt{N}$")
    ax.set_title("Convergence HMC")


    plt.show()


if __name__=="__main__":
    main()
    