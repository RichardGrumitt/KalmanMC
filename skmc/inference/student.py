import jax.numpy as jnp
from jaxopt import Bisection
from jax.scipy.special import digamma


# Jax modification of the pocoMC fitting routine.
# See https://github.com/minaskar/pocomc/blob/main/pocomc/student.py for original fitting function.


def fit_mvstud(data, tolerance=1e-6, max_iter=100):
    """
    Fit a multivariate Student's t distribution to data using the EM algorithm.

    Parameters
    ----------
    data : ndarray
        An array of shape (dim, n) containing n samples of dimension dim.
    tolerance : float, optional
        The tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. The default is 100.
    
    Returns
    -------
    mu : ndarray
        The mean of the distribution.
    Sigma : ndarray
        The covariance matrix of the distribution.
    nu : float
        The degrees of freedom of the distribution.
    """
    def opt_nu(delta_iobs, nu):
        def func0(nu):
            w_iobs = (nu + dim) / (nu + delta_iobs)
            f = -digamma(nu/2) + jnp.log(nu/2) + jnp.sum(jnp.log(w_iobs))/n - jnp.sum(w_iobs)/n + 1 + digamma((nu+dim)/2) - jnp.log((nu+dim)/2)
            return f

        if func0(1e6) >= 0:
            nu = jnp.inf
        else:
            nu, _ = Bisection(func0, 1e-6, 1e6).run()
        return nu

    data = data.T
    (dim, n) = data.shape
    mu = jnp.array([jnp.median(data, 1)]).T
    Sigma = jnp.cov(data) * (n-1) / n + 1e-1 * jnp.eye(dim)
    nu = 20

    last_nu = 0
    i = 0
    while jnp.abs(last_nu - nu) > tolerance and i < max_iter:
        i += 1
        diffs = data - mu
        delta_iobs = jnp.sum(diffs * jnp.linalg.solve(Sigma,diffs), 0)
        
        # update nu
        last_nu = nu
        try:
            nu = opt_nu(delta_iobs, nu)
        except Exception:
            pass

        if nu == jnp.inf:
            return mu.T[0], Sigma, nu

        w_iobs = (nu + dim) / (nu + delta_iobs)

        # update Sigma
        Sigma = jnp.dot(w_iobs*diffs, diffs.T) / n

        # update mu
        mu = jnp.sum(w_iobs * data, 1) / sum(w_iobs)
        mu = jnp.array([mu]).T

    return mu.T[0], Sigma, nu