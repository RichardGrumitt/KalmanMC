import jax
import jax.numpy as jnp
from tqdm import trange
import optax
import equinox as eqx
from typing import Callable, Tuple
from jaxtyping import Array, PRNGKeyArray


# Modification of FlowMC training routines to allow for monitoring of validation loss.
# See https://github.com/kazewong/flowMC/blob/main/src/flowMC/nfmodel/base.py for original training routines.


def make_training_loop(optim: optax.GradientTransformation) -> Callable:
    """
    Create a function that trains an NF model.

    Args:
        model (eqx.Model): NF model to train.
        optim (optax.GradientTransformation): Optimizer.

    Returns:
        train_flow: Function that trains the model.
    """

    @eqx.filter_value_and_grad
    def loss_fn(model, x):
        return -jnp.mean(model.log_prob(x))

    @eqx.filter_jit
    def train_step(model, x, opt_state):
        """Train for a single step.

        Args:
            model (eqx.Model): NF model to train.
            x (Array): Training data.
            opt_state (optax.OptState): Optimizer state.

        Returns:
            loss (Array): Loss value.
            model (eqx.Model): Updated model.
            opt_state (optax.OptState): Updated optimizer state.
        """
        loss, grads = loss_fn(model, x)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train_epoch(rng, model, state, train_ds, batch_size):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                value, model, state = train_step(model, batch, state)
        else:
            value, model, state = train_step(model, train_ds, state)

        return value, model, state

    def train_flow(
        rng: PRNGKeyArray,
        model: eqx.Module,
        data: Array,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        val_frac: float = None,
        verbose: bool = True,
    ) -> Tuple[PRNGKeyArray, eqx.Module, Array]:
        """Train a normalizing flow model.

        Args:
            rng (PRNGKeyArray): JAX PRNGKey.
            model (eqx.Module): NF model to train.
            data (Array): Training data.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            val_frac (float): Fraction of data to use for validation.
            verbose (bool): Whether to print progress.

        Returns:
            rng (PRNGKeyArray): Updated JAX PRNGKey.
            model (eqx.Model): Updated NF model.
            loss_values (Array): Loss values.
        """
        loss_values = jnp.zeros(num_epochs)
        if val_frac is not None:
            if val_frac <= 0.0 or val_frac >= 1.0:
                raise ValueError('val_frac should be between 0 and 1.')
            rng, split_rng = jax.random.split(rng)
            perms = jax.random.permutation(rng, data.shape[0])
            data = data[perms, ...]
            num_val = int(val_frac * data.shape[0])
            val_data = data[:num_val, ...]
            train_data = data[num_val:, ...]
        if verbose:
            pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        else:
            pbar = range(num_epochs)
        best_model = model
        best_loss = 1e9
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, model, state = train_epoch(input_rng, model, state, train_data, batch_size)
            if val_frac is None:
                loss_values = loss_values.at[epoch].set(value)
            elif val_frac is not None:
                val_loss = loss_fn(model, val_data)[0]
                loss_values = loss_values.at[epoch].set(val_loss)
            if loss_values[epoch] < best_loss:
                best_model = model
                best_loss = loss_values[epoch]
            if verbose:
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        if val_frac is None:
                            pbar.set_description(f"Training NF, training loss: {value:.3f}")
                        else:
                            pbar.set_description(f"Training NF, training loss: {value:.3f}, validation loss: {val_loss:.3f}")
                else:
                    if val_frac is None:
                        pbar.set_description(f"Training NF, training loss: {value:.3f}")
                    else:
                        pbar.set_description(f"Training NF, training loss: {value:.3f}, validation loss: {val_loss:.3f}")

        return rng, best_model, state, loss_values

    return train_flow, train_epoch, train_step