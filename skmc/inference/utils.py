import jax.numpy as jnp
import jax
import math
from jax.scipy.special import logsumexp


SQRTEPS = math.sqrt(float(jnp.finfo(jnp.float32).eps))


class Pearson:
    def __init__(self, a):
        self.l = a.shape[0]
        self.am = a - jnp.sum(a, axis=0) / self.l
        self.aa = jnp.sum(self.am**2, axis=0) ** 0.5

    def get(self, b):
        bm = b - jnp.sum(b, axis=0) / self.l
        bb = jnp.sum(bm**2, axis=0) ** 0.5
        ab = jnp.sum(self.am * bm, axis=0)
        return jnp.abs(ab / (self.aa * bb))


def multinomial_resample(key,
                         x,
                         logw):
    w = jnp.exp(logw)
    w /= jnp.sum(w)
    resampling_indexes = jax.random.choice(key, a=jnp.arange(len(w)), shape=(x.shape[0],), p=w)
    return x[resampling_indexes]


def systematic_resample(size, 
                        weights, 
                        random_key=None):
    """
        Resample a new set of points from the weighted set of inputs
        such that they all have equal weight.

    Parameters
    ----------
    size : `int`
        Number of samples to draw.
    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.
    random_key : jax PRNGKey, optional
        Jax random key.

    Returns
    -------
    indeces : `~numpy.ndarray` with shape (nsamples,)
        Indices of the resampled array.

    Notes
    -----
    Implements the systematic resampling method.
    """

    if random_key is None:
        random_key = jax.random.PRNGKey(0)

    if abs(jnp.sum(weights) - 1.) > SQRTEPS:
        weights = jnp.array(weights) / jnp.sum(weights)

    positions = (jax.random.uniform(random_key) + jnp.arange(size)) / size

    j = 0
    cumulative_sum = weights[0]
    indeces = jnp.empty(size, dtype=int)
    for i in range(size):
        while positions[i] > cumulative_sum:
            j += 1
            cumulative_sum += weights[j]
        indeces = indeces.at[i].set(j)
    
    return indeces


def compute_ess(logw):
    logw_max = jnp.amax(logw)
    logw_normed = logw - logw_max

    weights = jnp.exp(logw_normed) / jnp.sum(jnp.exp(logw_normed))
    return 1.0 / jnp.sum(weights * weights) / len(weights)


def get_beta(log_likelihood,
             old_beta,
             ess_target):

    low_beta = old_beta
    up_beta = 2.0

    rN = int(len(log_likelihood) * ess_target)

    while up_beta - low_beta > 1e-6:
        new_beta = (low_beta + up_beta) / 2.0
        log_weights_un = (new_beta - old_beta) * log_likelihood
        log_weights = log_weights_un - logsumexp(log_weights_un)
        ESS = int(jnp.exp(-logsumexp(log_weights * 2)))

        if ESS == rN:
            break
        elif ESS < rN:
            up_beta = new_beta
        else:
            low_beta = new_beta

    if new_beta >= 1:
        new_beta = 1

    return new_beta, ESS / len(log_likelihood)
