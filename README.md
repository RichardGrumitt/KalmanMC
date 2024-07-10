# KalmanMC

KalmanMC provides Jax implementations of Sequential Kalman Monte Carlo (SKMC) samplers for gradient-free inference in Bayesian inverse problems.

SKMC replaces the importance resampling step of standard Sequential Monte Carlo (SMC) with an Ensemble Kalman Inversion (EKI) update, before performing MCMC updates at each temperature level. For the MCMC updates we use the t-preconditioned Crank-Nicholson (tpCN) algorithm for efficient gradient-free sampling.   

The MCMC steps correct for the Gaussianity and model linearity assumptions of EKI, whilst accelerating the convergence of SMC. We also implement normalizing flow (NF) preconditioning, which can further improve performance on targets with nonlinear curvature.

The algorithms implemented in this repo are:

1. **NF-SKMC**: SKMC with NF preconditioning (faki_tpcn.py).
2. **SKMC**: SKMC without NF preconditioning (eki_tpcn.py).
3. **NF-SMC**: SMC with NF preconditioning (fis_tpcn.py).
4. **SMC**: SMC without NF preconditioning (is_tpcn.py).
5. **FAKI**: Flow Annealed Kalman Inversion i.e., temperature annealed EKI with NF preconditioning (faki.py).
6. **EKI**: temperature annealed EKI (eki.py).

For examples demonstrating the use of these algorithms on benchmark inverse problems see the `notebooks` directory.
