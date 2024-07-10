import jax
import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax.scipy.special import logsumexp

from .utils import Pearson, systematic_resample, get_beta
from .scaler import Reparameterise
from .student import fit_mvstud
from .training import make_training_loop

from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import equinox as eqx
import optax

from copy import deepcopy


class FIStpCN:
    def __init__(self,
                 log_like_fn: callable,
                 log_prior_fn: callable,
                 noise_cov: jnp.ndarray,
                 forward_map: callable,
                 data: jnp.ndarray,
                 scale: bool = True,
                 diagonal: bool = True,
                 flow_config: dict = None,
                 train_config: dict = None):
        """

        :param log_like_fn: function for evaluating the log likelihood.
        :param log_prior_fn: function for evaluating the log prior.
        :param noise_cov: observation noise covariance (just pass a dummy value for non-Gaussian likelihoods).
        :param forward_map: function for evaluating the forward model at given sample positions.
        :param data: observed data.
        :param scale: whether to scale variables to zero mean, unit variance.
        :param diagonal: whether to use diagonal covariance matrix for rescaling.
        :param flow_config: MAF configuration dictionary.
        :param train_config: MAF training dictionary.
        """
        self.log_like_fn = log_like_fn
        self.log_prior_fn = log_prior_fn
        self.noise_cov = noise_cov
        self.forward_map = forward_map
        self.data = data
        self.data_dim = noise_cov.shape[0]
        self.scale = scale
        self.diagonal = diagonal

        if flow_config is not None:
            self.flow_config = flow_config
        elif flow_config is None:
            self.flow_config = {'flow_type': 'MaskedCouplingRQSpline',
                                'n_layers': 2,
                                'nn_depth': 1,
                                'hidden_features': [50, 50, 50],
                                'n_bins': 4}
        if train_config is not None:
            self.train_config = train_config
        elif train_config is None:
            self.train_config = {'learning_rate': 1e-4,
                                 'momentum': 0.9,
                                 'validation_split': 0.1,
                                 'epochs': 200,
                                 'batch_size': 64}

        self.lr = train_config['learning_rate']
        self.momentum = train_config['momentum']
        self.val_frac = train_config['validation_split']
        self.n_epochs = train_config['epochs']
        self.batch_size = train_config['batch_size']

        if flow_config['flow_type'] != 'RealNVP' and flow_config['flow_type'] != 'MaskedCouplingRQSpline':
            raise ValueError('flow_type must be one of RealNVP or MaskedCouplingRQSpline')
        self.flow_type = flow_config['flow_type']
        self.n_layers = flow_config['n_layers']
        self.nn_depth = flow_config['nn_depth']
        self.n_hiddens = flow_config['hidden_features']
        self.n_bins = flow_config['n_bins']

    def run_FIStpCN_smc(self,
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
            k1, k2, k3, k4, key = jax.random.split(key, 5)

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
            
            self.setup_flow(x, k1)
            self.scaler = Reparameterise(n_dim=num_dim,
                                         scale=self.scale,
                                         diagonal=self.diagonal)
            self.scaler.fit(x)
            u = self.train_flow(x, k2)

            logw = (beta - old_beta) * log_like
            logw = logw - logsumexp(logw)
            resamp_idx = systematic_resample(num_ens, jnp.exp(logw), random_key=k3)
            x = x[resamp_idx]
            
            iters += 1
            ensembles[f'{iters}'] = deepcopy(x)

            u = self.scaler.forward(x)
            theta, J_flow = jax.vmap(lambda ui: self.flow.forward(ui))(u)
            J_flow = -1.0 * J_flow
            J = self.scaler.inverse(u)[1]

            log_like = self.log_like_fn(x)
            log_prior = self.log_prior_fn(x)

            state_dict = dict(theta=theta,
                              u=u,
                              x=x,
                              J=J,
                              J_flow=J_flow,
                              L=log_like,
                              P=log_prior,
                              ensembles=ensembles,
                              iters=iters,
                              beta=beta)

            option_dict = dict(target_accept=target_accept,
                               nmax=nmax,
                               patience=mcmc_patience,
                               correlation_threshold=correlation_threshold,
                               dim_threshold=dim_threshold,
                               sigma=sigma,
                               key=k4)

            pcn_result = self.nf_tpcn(state_dict,
                                      option_dict)
            x = pcn_result['x']
            accept = pcn_result['accept']
            sigma = pcn_result['efficiency']
            pcn_steps = pcn_result['steps']
            total_calls += pcn_result['calls']
            ensembles = pcn_result['ensembles']
            iters = pcn_result['iters']
            
            print(f'beta: {beta}, pCN steps: {pcn_steps}, pCN acceptance rate: {accept}, sigma: {sigma}')

        print(f'Total calls: {total_calls}')

        if return_verbose is True:
            return x, n, total_calls, ensembles
        else:
            return x

    def nf_tpcn(self,
                state_dict: dict,
                option_dict: dict):

        # Likelihood call counter
        n_calls = 0

        # Clone state variables
        theta = jnp.copy(state_dict.get('theta'))
        u = jnp.copy(state_dict.get('u'))
        x = jnp.copy(state_dict.get('x'))
        J = jnp.copy(state_dict.get('J'))
        J_flow = jnp.copy(state_dict.get('J_flow'))
        L = jnp.copy(state_dict.get('L'))
        P = jnp.copy(state_dict.get('P'))
        beta = state_dict.get('beta')

        key = option_dict.get('key')
        ensembles = state_dict.get('ensembles')
        iters = state_dict.get('iters')
        gamma_key, norm_key, unif_key = jax.random.split(key, 3)
        
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
        
        mu, cov, nu = fit_mvstud(theta)
        if ~jnp.isfinite(nu):
            nu = 1e6

        inv_cov = jnp.linalg.inv(cov)
        chol_cov = jnp.linalg.cholesky(cov)

        old_corr = 2
        corr = Pearson(x)

        i = 0
        while True:
            i += 1

            diff = theta - mu
            
            igamma_key, gamma_key = jax.random.split(gamma_key)
            gkeys = jax.random.split(igamma_key, n_walkers)
            s = jax.vmap(lambda k, d: 1.0 / (
                    2.0 / (nu + jnp.dot(d, jnp.dot(inv_cov, d))) * jax.random.gamma(k, (n_dim + nu) / 2)))(gkeys, diff)

            # Propose new points in theta space
            inorm_key, norm_key = jax.random.split(norm_key)
            nkeys = jax.random.split(norm_key, n_walkers)
            theta_prime = jax.vmap(lambda k, d, si: mu + (1.0 - sigma ** 2.0) ** 0.5 * d 
                                   + sigma * jnp.sqrt(si) * jnp.dot(chol_cov, jax.random.normal(k, shape=(n_dim,))))(nkeys, diff, s)

            # Transform to u space
            u_prime, J_flow_prime = self.flow.inverse(theta_prime)

            # Transform to x space
            x_prime, J_prime = self.scaler.inverse(u_prime)

            # Compute log-likelihood, log-prior, and log-posterior
            iunif_key, unif_key = jax.random.split(unif_key)
            u_rand = jax.random.uniform(iunif_key, shape=(n_walkers,))
    
            L_prime = self.log_like_fn(x_prime)
            P_prime = self.log_prior_fn(x_prime)

            n_calls += len(x_prime)

            # Compute Metropolis factors
            diff_prime = theta_prime - mu
            
            A = jax.vmap(lambda d: -(n_dim + nu) / 2 * jnp.log(1 + jnp.dot(d, jnp.dot(inv_cov, d)) / nu))(diff_prime)
            B = jax.vmap(lambda d: -(n_dim + nu) / 2 * jnp.log(1 + jnp.dot(d, jnp.dot(inv_cov, d)) / nu))(diff)
            alpha = jnp.minimum(
                jnp.ones(n_walkers),
                jnp.exp(L_prime * beta - L * beta + P_prime - P + J_prime - J + J_flow_prime - J_flow - A + B)
            )
            alpha = alpha.at[jnp.isnan(alpha)].set(0.0)

            # Metropolis criterion
            mask = u_rand < alpha

            # Accept new points            
            theta = theta.at[mask].set(theta_prime[mask])
            u = u.at[mask].set(u_prime[mask])
            x = x.at[mask].set(x_prime[mask])
            J = J.at[mask].set(J_prime[mask])
            J_flow = J_flow.at[mask].set(J_flow_prime[mask])
            L = L.at[mask].set(L_prime[mask])
            P = P.at[mask].set(P_prime[mask])

            iters += 1
            ensembles[f'{iters}'] = deepcopy(x)
            
            # Adapt scale parameter using diminishing adaptation
            sigma = jnp.exp(jnp.log(sigma) + (jnp.mean(alpha) - target_accept) / (i + 1))
            sigma = jnp.clip(sigma, 0.01, 1.0)

            # Adapt mean parameter using diminishing adaptation
            mu = mu + 1.0 / (i + 1.0) * (jnp.mean(theta, axis=0) - mu)

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

        return dict(theta=theta,
                    u=u, 
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

    def setup_flow(self, x, key):
        num_dim = x.shape[1]

        self.scaler = Reparameterise(n_dim=num_dim,
                                     scale=self.scale,
                                     diagonal=self.diagonal)
        self.scaler.fit(x)
        if self.flow_type == 'MaskedCouplingRQSpline':
            self.flow = MaskedCouplingRQSpline(num_dim,
                                               self.n_layers,
                                               self.n_hiddens,
                                               self.n_bins,
                                               key)
        elif self.flow_type == 'RealNVP':
            self.flow = RealNVP(num_dim,
                                self.n_layers,
                                self.n_hiddens,
                                key)

        self.tx = optax.chain(optax.clip(1.0), optax.adam(self.lr, self.momentum))
        self.training_loop, _, _ = make_training_loop(self.tx)
        self.optim_state = self.tx.init(eqx.filter(self.flow, eqx.is_array))

    def train_flow(self, x_train, flow_key):

        u_train = self.scaler.forward(x_train)
        key, self.flow, self.optim_state, loss_values = self.training_loop(flow_key,
                                                                           self.flow,
                                                                           u_train,
                                                                           self.optim_state,
                                                                           self.n_epochs,
                                                                           self.batch_size,
                                                                           self.val_frac,
                                                                           verbose=True)

        return u_train


