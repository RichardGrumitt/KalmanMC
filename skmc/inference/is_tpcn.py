import jax
import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax.scipy.special import logsumexp

from .utils import Pearson, systematic_resample, get_beta
from .scaler import Reparameterise
from .student import fit_mvstud

from copy import deepcopy


class IStpCN:
    def __init__(self,
                 log_like_fn: callable,
                 log_prior_fn: callable,
                 noise_cov: jnp.ndarray,
                 forward_map: callable,
                 data: jnp.ndarray,
                 scale: bool = True,
                 diagonal: bool = True):
        """

        :param log_like_fn: function for evaluating the log likelihood.
        :param log_prior_fn: function for evaluating the log prior.
        :param noise_cov: observation noise covariance (just pass a dummy value for non-Gaussian likelihoods).
        :param forward_map: function for evaluating the forward model at given sample positions.
        :param data: observed data.
        :param scale: whether to scale variables to zero mean, unit variance.
        :param diagonal: whether to use diagonal covariance matrix for rescaling.
        """
        self.log_like_fn = log_like_fn
        self.log_prior_fn = log_prior_fn
        self.noise_cov = noise_cov
        self.forward_map = forward_map
        self.data = data
        self.data_dim = noise_cov.shape[0]
        self.scale = scale
        self.diagonal = diagonal

    def run_IStpCN_smc(self,
                       prior_samples: jnp.ndarray,
                       beta_schedule: jnp.ndarray = None,
                       target_accept: float = 0.5,
                       nmax: int = 25,
                       mcmc_patience: int = 10,
                       ess_target: float = 0.5,
                       correlation_threshold: float = 0.01,
                       dim_threshold: float = 0.9,
                       sigma: float = None,
                       seed: int = 0,
                       return_verbose: bool = False):

        x = deepcopy(prior_samples)

        key = jax.random.PRNGKey(seed)

        iters = 0
        n = 0
        beta = 0.0
        total_calls = 0
        num_ens = x.shape[0]
        num_dim = x.shape[1]
        
        if sigma is None:
            sigma = jnp.minimum(2.38 / num_dim ** 0.5, 0.99)
            
        ensembles = {f'{iters}': deepcopy(prior_samples)}

        while beta < 1.0:

            n += 1
            k1, k2, key = jax.random.split(key, 3)

            forward_eval = self.forward_map(x)
            lsq_vector = jax.vmap(lambda f: (self.data - f).T @ inv(self.noise_cov) @ (self.data - f))(forward_eval)
            log_like = -0.5 * lsq_vector
            if beta_schedule is None:
                old_beta = deepcopy(beta)
                beta, _ = get_beta(log_like, beta, ess_target)
                alpha = 1.0 / (beta - old_beta)
            else:
                old_beta = deepcopy(beta)
                alpha = 1 / (beta_schedule[n - 1] - beta)
                beta = beta_schedule[n - 1]
            print(f'Iteration {n}, beta: {beta}, alpha: {alpha}')
            
            self.scaler = Reparameterise(n_dim=num_dim,
                                         scale=self.scale,
                                         diagonal=self.diagonal)
            self.scaler.fit(x)
            u = self.scaler.forward(x)

            t_mu, t_cov, t_nu = fit_mvstud(u)
            if ~jnp.isfinite(t_nu):
                t_nu = 1e6

            logw = (beta - old_beta) * log_like
            logw = logw - logsumexp(logw)
            resamp_idx = systematic_resample(num_ens, jnp.exp(logw), random_key=k1)
            x = x[resamp_idx]
            
            iters += 1
            ensembles[f'{iters}'] = deepcopy(x)

            u = self.scaler.forward(x)
            J = self.scaler.inverse(u)[1]
            
            t_mu, t_cov, t_nu = fit_mvstud(u)
            if ~jnp.isfinite(t_nu):
                t_nu = 1e6

            log_like = self.log_like_fn(x)
            log_prior = self.log_prior_fn(x)

            state_dict = dict(u=u,
                              x=x,
                              J=J,
                              L=log_like,
                              P=log_prior,
                              ensembles=ensembles,
                              iters=iters,
                              beta=beta,
                              mu=t_mu,
                              cov=t_cov,
                              nu=t_nu)

            option_dict = dict(target_accept=target_accept,
                               nmax=nmax,
                               patience=mcmc_patience,
                               correlation_threshold=correlation_threshold,
                               dim_threshold=dim_threshold,
                               sigma=sigma,
                               key=k2)

            pcn_result = self.tpcn(state_dict,
                                   option_dict)
            x = pcn_result['x']
            accept = pcn_result['accept']
            sigma = pcn_result['efficiency']
            pcn_steps = pcn_result['steps']
            total_calls += pcn_result['calls']
            ensembles = pcn_result['ensembles']
            iters = pcn_result['iters']
            
            print(f'beta: {beta}, pCN steps: {pcn_steps}, pCN acceptance rate: {accept}, efficiency: {sigma}')

        print(f'Total calls: {total_calls}')

        if return_verbose is True:
            return x, n, total_calls, ensembles
        else:
            return x

    def tpcn(self,
             state_dict: dict,
             option_dict: dict):

        # Likelihood call counter
        n_calls = 0

        # Clone state variables
        u = jnp.copy(state_dict.get('u'))
        x = jnp.copy(state_dict.get('x'))
        J = jnp.copy(state_dict.get('J'))
        L = jnp.copy(state_dict.get('L'))
        P = jnp.copy(state_dict.get('P'))
        beta = state_dict.get('beta')
        key = option_dict.get('key')
        ensembles = state_dict.get('ensembles')
        iters = state_dict.get('iters')
        gamma_key, norm_key, unif_key = jax.random.split(key, 3)
        mu = state_dict.get('mu')
        cov = state_dict.get('cov')
        nu = state_dict.get('nu')
        
        # Get MCMC options
        n_max = option_dict.get('nmax')
        progress_bar = option_dict.get('progress_bar')
        target_accept = option_dict.get('target_accept')
        sigma = option_dict.get('sigma')
        patience = option_dict.get('patience')
        correlation_threshold = option_dict.get('correlation_threshold')
        dim_threshold = option_dict.get('dim_threshold')

        # Get number of particles and parameters/dimensions
        n_walkers, n_dim = x.shape

        inv_cov = jnp.linalg.inv(cov)
        chol_cov = jnp.linalg.cholesky(cov)

        old_corr = 2
        corr = Pearson(x)

        i = 0
        while True:
            i += 1

            diff = u - mu
            
            igamma_key, gamma_key = jax.random.split(gamma_key)
            gkeys = jax.random.split(igamma_key, n_walkers)
            s = jax.vmap(lambda k, d: 1.0 / (
                    2.0 / (nu + jnp.dot(d, jnp.dot(inv_cov, d))) * jax.random.gamma(k, (n_dim + nu) / 2)))(gkeys, diff)

            # Propose new points in u space
            inorm_key, norm_key = jax.random.split(norm_key)
            nkeys = jax.random.split(norm_key, n_walkers)
            u_prime = jax.vmap(lambda k, d, si: mu + (1.0 - sigma ** 2.0) ** 0.5 * d 
                                + sigma * jnp.sqrt(si) * jnp.dot(chol_cov, jax.random.normal(k, shape=(n_dim,))))(nkeys, diff, s)
            
            # Transform to x space
            x_prime, J_prime = self.scaler.inverse(u_prime)

            # Compute log-likelihood, log-prior, and log-posterior
            iunif_key, unif_key = jax.random.split(unif_key)
            u_rand = jax.random.uniform(iunif_key, shape=(n_walkers,))
    
            L_prime = self.log_like_fn(x_prime)
            P_prime = self.log_prior_fn(x_prime)

            n_calls += len(x_prime)

            # Compute Metropolis factors
            diff_prime = u_prime - mu
            
            A = jax.vmap(lambda d: -(n_dim + nu) / 2 * jnp.log(1 + jnp.dot(d, jnp.dot(inv_cov, d)) / nu))(diff_prime)
            B = jax.vmap(lambda d: -(n_dim + nu) / 2 * jnp.log(1 + jnp.dot(d, jnp.dot(inv_cov, d)) / nu))(diff)
            alpha = jnp.minimum(
                jnp.ones(n_walkers),
                jnp.exp(L_prime * beta - L * beta + P_prime - P + J_prime - J - A + B)
            )
            alpha = alpha.at[jnp.isnan(alpha)].set(0.0)

            # Metropolis criterion
            mask = u_rand < alpha

            # Accept new points
            u = u.at[mask].set(u_prime[mask])
            x = x.at[mask].set(x_prime[mask])
            J = J.at[mask].set(J_prime[mask])
            L = L.at[mask].set(L_prime[mask])
            P = P.at[mask].set(P_prime[mask])
            
            iters += 1
            ensembles[f'{iters}'] = deepcopy(x)

            # Adapt scale parameter using diminishing adaptation
            sigma = jnp.exp(jnp.log(sigma) + (jnp.mean(alpha) - target_accept) / (i + 1))
            sigma = jnp.clip(sigma, 0.01, 1.0)

            # Adapt mean parameter using diminishing adaptation
            mu = mu + 1.0 / (i + 1.0) * (jnp.mean(u, axis=0) - mu)

            # Update progress bar if available
            if progress_bar is not None:
                progress_bar.update_stats(
                    dict(calls=progress_bar.info['calls'] + n_walkers,
                         acc=jnp.mean(alpha),
                         steps=i,
                         logP=jnp.mean(L + P),
                         eff=sigma / (2.38 / jnp.sqrt(n_dim)),
                         )
                 )

            # Loop termination criteria:
            pearson_r = corr.get(x)
            if jnp.mean((old_corr - pearson_r) > correlation_threshold) > dim_threshold:
                old_corr = pearson_r
            elif jnp.mean((old_corr - pearson_r) > correlation_threshold) <= dim_threshold and i >= patience:
                break

            if i >= n_max:
                break

        return dict(u=u, 
                    x=x, 
                    J=J, 
                    L=L, 
                    P=P,
                    efficiency=sigma, 
                    accept=jnp.mean(alpha), 
                    steps=i, 
                    calls=n_calls,
                    ensembles=ensembles,
                    iters=iters,
                    )

