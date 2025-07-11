from typing import Callable

from  jax import vmap
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, constant, zeros, uniform


from pinn.embeddings import FourierEmbedding, WaveletEmbedding
from pinn.activation import Snake, ModifiedReLU


def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    if name == "snake":
        return Snake()
    elif name == "modified_relu":
        return ModifiedReLU()
    else:
        return getattr(nn, name, nn.tanh)  # Default to tanh if not found


class Dense(nn.Module):
    in_features: int
    out_features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = normal(0.1)

    def setup(self):
        self.kernel = self.param(
            "kernel", self.kernel_init, (self.in_features, self.out_features)
        )
        self.bias = self.param("bias", self.bias_init, (self.out_features,))

    @nn.compact
    def __call__(self, x):
        return jnp.dot(x, self.kernel) + self.bias


class MLPBlock(nn.Module):
    hidden_dim: int
    num_layers: int
    act_fn: Callable
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
        return Dense(x.shape[-1], self.out_dim or x.shape[-1])(x)



class MLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = 2.0
    emb_dim: int = 64

    def setup(self):
        self.act_fn = get_activation(self.act_name)
        self.sl = self.param("sl", constant(0.0), (1,))

    @nn.compact
    def __call__(self, x):

        if self.fourier_emb:
            x = FourierEmbedding(emb_scale=self.emb_scale, emb_dim=self.emb_dim)(x)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
        return Dense(x.shape[-1], self.out_dim)(x)


class ResNet(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = 2.0
    emb_dim: int = 64

    def setup(self):
        self.act_fn = get_activation(self.act_name)
        self.sl = self.param("sl", constant(0.0), (1,))

    @nn.compact
    def __call__(self, x):

        if self.fourier_emb:
            x = FourierEmbedding(emb_scale=self.emb_scale, emb_dim=self.emb_dim)(x)

        x = Dense(x.shape[-1], self.hidden_dim)(x)
        for _ in range(self.num_layers):
            x_res = x
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
            x = x + x_res
        return Dense(self.hidden_dim, self.out_dim)(x)


class ModifiedMLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = 2.0
    emb_dim: int = 64

    def setup(self):
        self.act_fn = get_activation(self.act_name)
        self.sl = self.param("sl", constant(0.0), (1,))


    @nn.compact
    def __call__(self, x, t):

        if self.fourier_emb:
            x = FourierEmbedding(emb_scale=self.emb_scale, emb_dim=self.emb_dim)(x)

        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            # x = self.act_fn(x)
            x = nn.tanh(x)
            x = x * u + (1 - x) * v

        return Dense(x.shape[-1], self.out_dim)(x)

