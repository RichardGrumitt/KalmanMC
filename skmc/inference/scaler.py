import numpy as np
import jax.numpy as jnp


class Reparameterise:
    """
    Class that reparameterises the model using change-of-variables parameter transformations.

    Parameters
    ----------
    n_dim : ``int``
        Dimensionality of sampling problem
    scale : ``bool``
        Rescale parameters to zero mean and unit variance (default is true)
    diagonal : ``bool``
        Use diagonal transformation (i.e. ignore covariance) (default is true)
    """
    def __init__(self,
                 n_dim: int,
                 scale: bool = True,
                 diagonal: bool = True):

        self.ndim = n_dim

        self.mu = None
        self.sigma = None
        self.cov = None
        self.L = None
        self.L_inv = None
        self.log_det_L = None
        self.scale = scale
        self.diagonal = diagonal

    def fit(self, x: np.ndarray):
        """
        Learn mean and standard deviation using for rescaling.
        
        Parameters
        ----------
        x : np.ndarray
            Input data used for training.
        """
        u = jnp.copy(x)
        self.mu = jnp.mean(u, axis=0)
        if self.diagonal:
            self.sigma = jnp.std(u, axis=0)
        else:
            self.cov = jnp.cov(u.T)
            self.L = jnp.linalg.cholesky(self.cov)
            self.L_inv = jnp.linalg.inv(self.L)
            self.log_det_L = jnp.linalg.slogdet(self.L)[1]

    def forward(self, x: np.ndarray):
        """
        Forward transformation (both logit/probit for bounds and affine for all parameters).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """

        u = jnp.copy(x)
        if self.scale:
            u = self._forward_affine(u)
        return u

    def inverse(self, u: np.ndarray):
        """
        Inverse transformation (both logit^-1/probit^-1 for bounds and affine for all parameters).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        if self.scale:
            x, log_det_J = self._inverse_affine(u)
        else:
            x, log_det_J = jnp.copy(u), jnp.zeros_like(len(u))

        return x, log_det_J

    def _forward_affine(self, x: np.ndarray):
        """
        Forward affine transformation.

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        if self.diagonal:
            return (x - self.mu) / self.sigma
        else:
            return jnp.array([jnp.dot(self.L_inv, xi - self.mu) for xi in x])

    def _inverse_affine(self, u: np.ndarray):
        """
        Inverse affine transformation.

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.ndarray
            Diagonal of Jacobian matrix.
        """
        if self.diagonal:
            J = self.sigma
            log_det_J = jnp.linalg.slogdet(J * jnp.identity(len(J)))[1]
            return self.mu + self.sigma * u, log_det_J * jnp.ones(len(u))
        else:
            x = self.mu + jnp.array([jnp.dot(self.L, ui) for ui in u])
            return x, self.log_det_L * jnp.ones(len(u))
