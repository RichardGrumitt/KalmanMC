import jax
import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import inv

from .utils import get_beta
from .scaler import Reparameterise
from .training import make_training_loop
import optax
from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import equinox as eqx

from copy import deepcopy


class FAKI:
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

    def faki_update(self,
                    alpha: float,
                    prior_samples: jnp.ndarray,
                    forward_eval: jnp.ndarray,
                    key: jax.random.PRNGKey):

        num_ens = prior_samples.shape[0]
        num_dim = prior_samples.shape[1]
        x = deepcopy(prior_samples)
        k1, k2, k3 = key
            
        self.setup_flow(x, k1)
        self.scaler = Reparameterise(n_dim=num_dim,
                                     scale=self.scale,
                                     diagonal=self.diagonal)
        self.scaler.fit(x)
        u = self.train_flow(x, k2)
        theta, log_det_J = jax.vmap(lambda ui: self.flow.forward(ui))(u)
        theta_mean = jnp.mean(theta, axis=0)
            
        forward_mean = jnp.mean(forward_eval, axis=0)

        Z_p_t = theta - theta_mean
        Y_p_t = forward_eval - forward_mean
        cov_ty = jnp.sum(jax.vmap(lambda z, y: z.reshape(-1, 1) @ y.reshape(-1, 1).T)(Z_p_t, Y_p_t), axis=0) / (
                num_ens - 1.0)
        cov_yy = jnp.sum(jax.vmap(lambda y: y.reshape(-1, 1) @ y.reshape(-1, 1).T)(Y_p_t), axis=0) / (
                num_ens - 1.0) + self.noise_cov * alpha
        Q = cov_ty @ inv(cov_yy)

        obs_noise = jax.random.multivariate_normal(k3, mean=jnp.zeros(self.noise_cov.shape[0]), cov=self.noise_cov,
                                                   shape=(num_ens,))
        theta_prime = jax.vmap(lambda t, n, y: t + Q @ (self.data + np.sqrt(alpha) * n - y))(theta, obs_noise,
                                                                                             forward_eval)

        u_prime, _ = self.flow.inverse(theta_prime)
        x_prime, _ = self.scaler.inverse(u_prime)

        return x_prime

    def run_faki(self,
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
            
        ensembles = {}
        ensembles[f'{iters}'] = deepcopy(prior_samples)
        
        while beta < 1.0:

            n += 1
            k1, k2, k3, key = jax.random.split(key, 4)

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

            x = self.faki_update(alpha=alpha,
                                 prior_samples=deepcopy(x),
                                 forward_eval=deepcopy(forward_eval),
                                 key=(k1, k2, k3))
            total_calls += x.shape[0]
            iters += 1
            ensembles[f'{iters}'] = deepcopy(x)

        print(f'Total calls: {total_calls}')

        if return_verbose is True:
            return x, n, total_calls, ensembles
        else:
            return x

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
