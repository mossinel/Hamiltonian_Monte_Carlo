# Hamiltonian Monte Carlo.

Implementation and application of an Hamiltonian Monte Carlo method to sample from unnormalized distribution.

Team members: Maxence Hofer, Giacomo Mossinelli



# Overview

`figure/` - folder containing all figures produced

`src/` - scripts for every simulation


### Code

- `d.py` - functions used in point d)

- `graph_density.py` - produce the density graphs.

- `graphs_autocorrelation_single.py` - produce the average and variance, autocovariance, autocorrelation and distribution of effective sample size

- `ess_single.py` - produce the effective sample size for the number of evaluation of f and df/dq

- `e_new_new.py` - functions used in point e)

- `ess_e.py` - produce the effective sample size graph for point e)

- `convergence_analysis.py` - produce the comparison graphs between HMC and 1MH for point f)



# Environment

We use `Python 3.10.9`, `NumPy`, `scipy.stats` and `matplotlib.pyplot`.
