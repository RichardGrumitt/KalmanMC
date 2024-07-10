import jax
import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import inv
from .utils import get_beta

from copy import deepcopy


class EKI:
    def __init__(self,
                 log_like_fn: callable,
                 log_prior_fn: callable,
                 noise_cov: jnp.ndarray,
                 forward_map: callable,
                 data: jnp.ndarray):
        """

        :param log_like_fn: function for evaluating the log likelihood.
        :param log_prior_fn: function for evaluating the log prior.
        :param noise_cov: observation noise covariance (just pass a dummy value for non-Gaussian likelihoods).
        :param forward_map: function for evaluating the forward model at given sample positions.
        :param data: observed data.
        """
        self.log_like_fn = log_like_fn
        self.log_prior_fn = log_prior_fn
        self.noise_cov = noise_cov
        self.forward_map = forward_map
        self.data = data

    def eki_update(self,
                   alpha: float,
                   prior_samples: jnp.ndarray,
                   forward_eval: jnp.ndarray,
                   key: jax.random.PRNGKey):

        num_ens = prior_samples.shape[0]

        theta = deepcopy(prior_samples)
        theta_mean = jnp.mean(theta, axis=0)

        forward_mean = jnp.mean(forward_eval, axis=0)

        Z_p_t = theta - theta_mean
        Y_p_t = forward_eval - forward_mean
        cov_ty = jnp.sum(jax.vmap(lambda z, y: z.reshape(-1, 1) @ y.reshape(-1, 1).T)(Z_p_t, Y_p_t), axis=0) / (
                num_ens - 1.0)
        cov_yy = jnp.sum(jax.vmap(lambda y: y.reshape(-1, 1) @ y.reshape(-1, 1).T)(Y_p_t), axis=0) / (
                num_ens - 1.0) + self.noise_cov * alpha
        Q = cov_ty @ inv(cov_yy)

        obs_noise = jax.random.multivariate_normal(key, mean=jnp.zeros(self.noise_cov.shape[0]), cov=self.noise_cov,
                                                   shape=(num_ens,))
        theta_prime = jax.vmap(lambda t, n, y: t + Q @ (self.data + np.sqrt(alpha) * n - y))(theta, obs_noise,
                                                                                             forward_eval)

        return theta_prime

    def run_eki(self,
                prior_samples: jnp.ndarray,
                beta_schedule: jnp.ndarray = None,
                ess_target: float = 0.5,
                seed: int = 0,
                return_verbose: bool = False):

        x = deepcopy(prior_samples)

        key = jax.random.PRNGKey(seed)

        iters = 0
        n = 0
        beta = 0.0
        total_calls = 0
        
        ensembles = {f'{iters}': deepcopy(prior_samples)}

        while beta < 1.0:

            n += 1
            new_key, key = jax.random.split(key)

            if beta_schedule is None:
                forward_eval = self.forward_map(x)
                lsq_vector = jax.vmap(lambda f: (self.data - f).T @ inv(self.noise_cov) @ (self.data - f))(forward_eval)
                old_beta = deepcopy(beta)
                beta, _ = get_beta(-0.5 * lsq_vector, beta, ess_target)
                alpha = 1.0 / (beta - old_beta)
            else:
                forward_eval = self.forward_map(x)
                alpha = 1 / (beta_schedule[n - 1] - beta)
                beta = beta_schedule[n - 1]

            print(f'Iteration {n}, beta: {beta}, alpha: {alpha}')

            x = self.eki_update(alpha=alpha,
                                prior_samples=deepcopy(x),
                                forward_eval=deepcopy(forward_eval),
                                key=new_key)
            total_calls += x.shape[0]
            iters += 1
            ensembles[f'{iters}'] = deepcopy(x)

        print(f'Total calls: {total_calls}')

        if return_verbose:
            return x, n, total_calls, ensembles
        else:
            return x
