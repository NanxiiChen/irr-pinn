from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import jit


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    xdim = kwargs.get("xdim", 3)
    params = model.init(rng, jnp.ones(xdim), jnp.ones(1))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.adam(scheduler)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jit, static_argnums=(0,))
def train_step(loss_fn, state, batch, eps):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(params, batch, eps)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)


class StaggerSwitch:
    def __init__(self, pde_names=["ac", "ch",], stagger_period=10):
        self.pde_names = pde_names
        self.stagger_period = stagger_period
        self.epoch = 0

    def step_epoch(self):
        self.epoch += 1

    def decide_pde(self):
        epoch_round = len(self.pde_names) * self.stagger_period
        idx = (self.epoch % epoch_round) // self.stagger_period
        return self.pde_names[idx]
    

if __name__ == "__main__":
    stagger_switch = StaggerSwitch()
    for i in range(50):
        print(i, stagger_switch.decide_pde())
        stagger_switch.step_epoch()

