import jax.numpy as jnp
from jax import random

from pinn import lhs_sampling, shifted_grid, Sampler


class CombustionSampler(Sampler):

    def __init__(
        self,
        n_samples,
        domain=((0, 1.0),),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples,],
            key,
        )
        return data
    
    def sample_bc(self):
        left, right = self.domain[0]
        left = jnp.array([left,])
        right = jnp.array([right,])
        return left, right
    
    def sample(self, *args, **kwargs):
        left, right = self.sample_bc()
        pde = self.sample_pde(),
        return (
            pde, # for PDE loss
            left, right,
            pde # for irreversible loss
        )