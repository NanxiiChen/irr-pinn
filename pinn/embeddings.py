from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, constant, zeros, uniform


class FourierEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 64

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel",
            normal(self.emb_scale),
            (x.shape[-1], self.emb_dim),
        )
        return jnp.concatenate(
            [
                jnp.sin(jnp.pi * jnp.dot(x, kernel)),
                jnp.cos(jnp.pi * jnp.dot(x, kernel)),
            ],
            axis=-1,
        )


class RBFEmbedding(nn.Module):
    emb_dim: int = 64
    emb_scale: float = 0.1
    emb_width: float = 0.05

    @nn.compact
    def __call__(self, x):

        centers = self.param(
            "kernel",
            normal(self.emb_scale),
            (self.emb_dim, x.shape[-1]),
        )  # --> shape (emb_dim, xdim)

        x = jnp.expand_dims(x, axis=0)
        dist_sq = jnp.sum((x - centers) ** 2, axis=-1)
        rbf = jnp.exp(-dist_sq / (2 * self.emb_width**2))
        return rbf


class ExponentialEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x):
        low, high = 1 / self.emb_scale, self.emb_scale

        def kernel_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype, minval=low, maxval=high)

        kernel = self.param("kernel", kernel_init, (x.shape[-1], self.emb_dim))
        x = jnp.dot(x, jnp.ones((x.shape[-1], self.emb_dim)))
        x = x**kernel
        return x.reshape(-1)


class WaveletEmbedding(nn.Module):
    levels: int = 4
    emb_dim: int = 64
    
    
    @nn.compact
    def __call__(self, x):
        coefs = []
        for level in range(self.levels):
            scale = 2**level
            kernel = self.param(
                f"kernel_{level}",
                normal(0.1 * scale),
                (x.shape[-1], self.emb_dim // self.levels)
            )
            scaled_x = x * scale
            wave = jnp.sin(jnp.pi * jnp.dot(scaled_x, kernel)) * jnp.exp(-0.5 * jnp.abs(scaled_x))
            coefs.append(wave)
            
        return jnp.concatenate(coefs, axis=-1)