
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, constant, zeros


class Dense(nn.Module):
    in_features: int
    out_features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = normal(0.1)

    def setup(self):
        self.kernel = self.param('kernel', self.kernel_init,
                                 (self.in_features, self.out_features))
        self.bias = self.param('bias', self.bias_init,
                               (self.out_features,))

    @nn.compact
    def __call__(self, x):
        return jnp.dot(x, self.kernel) + self.bias


class FourierEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 64

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', normal(self.emb_scale),
                            (x.shape[-1], self.emb_dim))
        return jnp.concatenate([jnp.sin(jnp.pi*jnp.dot(x, kernel)),
                                jnp.cos(jnp.pi*jnp.dot(x, kernel))], axis=-1)
        
class ExponentialEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 32
    
    @nn.compact
    def __call__(self, x):
        low, high = 1/self.emb_scale, self.emb_scale
        
        def kernel_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype, minval=low, maxval=high)
        
        kernel = self.param('kernel', kernel_init, (x.shape[-1], self.emb_dim))
        x = jnp.dot(x, jnp.ones((x.shape[-1], self.emb_dim)))
        x = x ** kernel
        return x.reshape(-1)
    


class MLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64
    
    def setup(self):
        self.act_fn = getattr(nn, self.act_name)

    @nn.compact
    def __call__(self, x, t):

        if self.fourier_emb:
            # separate the spatial and temporal coordinates
            t_emb = FourierEmbedding(
                emb_scale=self.emb_scale[1], 
                emb_dim=self.emb_dim)(t)
            x_emb = FourierEmbedding(
                emb_scale=self.emb_scale[0], 
                emb_dim=self.emb_dim)(x)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        for _ in range(self.num_layers - 1):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
        return Dense(x.shape[-1], self.out_dim)(x)



# class ModifiedMLPBlock(nn.Module):
#     hidden_dim: int
#     num_layers: int
#     act_fn: callable

#     @nn.compact
#     def __call__(self, x):
#         u = Dense(x.shape[-1], self.hidden_dim)(x)
#         v = Dense(x.shape[-1], self.hidden_dim)(x)
#         u = self.act_fn(u)
#         v = self.act_fn(v)
        
#         for _ in range(self.num_layers):
#             x = Dense(x.shape[-1], self.hidden_dim)(x)
#             x = nn.tanh(x)
#             x = x * u + (1 - x) * v
            
#         return x

# class ModifiedMLP(nn.Module):
#     act_name: str = "tanh"
#     num_layers: int = 4
#     hidden_dim: int = 64
#     out_dim: int = 2
#     fourier_emb: bool = True
#     emb_scale: tuple = (2.0, 2.0)
#     emb_dim: int = 64

#     def setup(self):
#         self.act_fn = getattr(nn, self.act_name)

#         self.spatial_block = ModifiedMLPBlock(
#             hidden_dim=self.hidden_dim,
#             num_layers=self.num_layers,
#             act_fn=self.act_fn
#         )
#         self.temporal_block = ModifiedMLPBlock(
#             hidden_dim=self.hidden_dim,
#             num_layers=self.num_layers,
#             act_fn=self.act_fn
#         )
#         self.output_layer = Dense(self.hidden_dim, self.out_dim)

#     @nn.compact
#     def __call__(self, x, t):
#         x_emb = FourierEmbedding(
#             emb_scale=self.emb_scale[0], 
#             emb_dim=self.emb_dim)(x)
#         t_emb = FourierEmbedding(
#             emb_scale=self.emb_scale[1], 
#             emb_dim=self.emb_dim)(t)
        
#         x_emb = Dense(x_emb.shape[-1], self.hidden_dim)(x_emb)
#         t_emb = Dense(t_emb.shape[-1], self.hidden_dim)(t_emb)
        
#         x_features = self.spatial_block(x_emb)
#         t_features = self.temporal_block(t_emb)
        
#         combined = (x_features + 1) * (t_features + 1) - 1
        
#         return self.output_layer(combined)




class ModifiedMLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64

    def setup(self):
        self.act_fn = getattr(nn, self.act_name)

    @nn.compact
    def __call__(self, x, t):

        if self.fourier_emb:
            t_emb = FourierEmbedding(
                emb_scale=self.emb_scale[1], 
                emb_dim=self.emb_dim)(t)
            x_emb = FourierEmbedding(
                emb_scale=self.emb_scale[0], 
                emb_dim=self.emb_dim)(x)
            
            # x_emb = Dense(x_emb.shape[-1], self.hidden_dim)(x_emb)
            # t_emb = Dense(t_emb.shape[-1], self.hidden_dim)(t_emb)
            # x = (1 + x_emb) * (1 + t_emb) - 1
            
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
            
            # x = FourierEmbedding(emb_scale=2.0)(jnp.concatenate([x, t], axis=-1))
        else:
            x = jnp.concatenate([x, t], axis=-1)

        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = nn.tanh(x)
            x = x * u + (1 - x) * v

        return Dense(x.shape[-1], self.out_dim)(x)
