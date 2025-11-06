from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, constant, zeros, uniform
from jaxkan.KAN import KAN
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




class MLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64

    def setup(self):
        self.act_fn = get_activation(self.act_name)

    @nn.compact
    def __call__(self, x, t):

        if self.fourier_emb:
            # separate the spatial and temporal coordinates
            t_emb = FourierEmbedding(emb_scale=self.emb_scale[1], emb_dim=self.emb_dim)(
                t
            )
            x_emb = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(
                x
            )
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

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
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64

    def setup(self):
        self.act_fn = get_activation(self.act_name)

    @nn.compact
    def __call__(self, x, t):
        if self.fourier_emb:
            if len(self.emb_scale) == 1:
                # x = jnp.concatenate([x, t], axis=-1)
                # x = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                t_emb = Dense(t.shape[-1], self.emb_dim)(t)
                x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                x = jnp.concatenate([x_emb, t_emb], axis=-1)
            else:
                t_emb = FourierEmbedding(self.emb_scale[1], self.emb_dim)(t)
                x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        x = Dense(x.shape[-1], self.hidden_dim)(x)
        for _ in range(self.num_layers):
            x_res = x
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
            x = x + x_res
        return Dense(self.hidden_dim, self.out_dim)(x)


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
        self.act_fn = get_activation(self.act_name)

    @nn.compact
    def __call__(self, x, t):
        if self.fourier_emb:
            if len(self.emb_scale) == 1:
                # x = jnp.concatenate([x, t], axis=-1)
                # x = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                t_emb = Dense(t.shape[-1], self.emb_dim)(t)
                x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                x = jnp.concatenate([x_emb, t_emb], axis=-1)
            else:
                t_emb = FourierEmbedding(self.emb_scale[1], self.emb_dim)(t)
                x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)


        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = nn.tanh(x)
            # x = self.act_fn(x)
            x = x * u + (1 - x) * v

        return Dense(x.shape[-1], self.out_dim)(x)



class ExpertMLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 6
    hidden_dim: int = 64
    out_dim: int = 3

    def setup(self):
        self.act_fn = get_activation(self.act_name)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
        return Dense(x.shape[-1], self.out_dim)(x)


class GatingNetwork(nn.Module):
    n_experts: int = 4
    hidden_dim: int = 64
    num_layers: int = 6
    act_name: str = "tanh"

    def setup(self):
        if self.act_name == "snake":
            self.act_fn = Snake()
        else:
            self.act_fn = getattr(nn, self.act_name)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)

        logits = Dense(x.shape[-1], self.n_experts)(x)

        return nn.softmax(logits, axis=-1)


class MixtureOfExperts(nn.Module):
    n_experts: int = 4
    hidden_dim: int = 64
    num_layers: int = 6
    out_dim: int = 3
    act_name: str = "tanh"
    fourier_emb: bool = False
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64

    def setup(self):
        self.act_fn = get_activation(self.act_name)

        self.experts = [ExpertMLP(
            act_name=self.act_name,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
        ) for _ in range(self.n_experts)]          

    @nn.compact
    def __call__(self, x, t):
        if self.fourier_emb:
            t_emb = FourierEmbedding(self.emb_scale[1], self.emb_dim)(t)
            x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        gating_weights = GatingNetwork(
            n_experts=self.n_experts,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            act_name=self.act_name,
        )(x)

        # 使用预先创建的专家模型
        expert_outputs = jnp.stack([expert(x) for expert in self.experts], axis=-1)

        output = jnp.sum(gating_weights[None, ...] * expert_outputs, axis=-1)
        return output


class PirateNet(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64
    nonlinearity: float = 0.5

    def setup(self):
        self.act_fn = get_activation(self.act_name)

    @nn.compact
    def __call__(self, x, t):
        if self.fourier_emb:
            t_emb = FourierEmbedding(self.emb_scale[1], self.emb_dim)(t)
            x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        x = Dense(x.shape[-1], self.hidden_dim)(x)
        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for idx in range(self.num_layers):
            identity = x

            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = nn.tanh(x)
            x = x * u + (1 - x) * v
            # x = Dense(x.shape[-1], self.hidden_dim)(x)
            # x = nn.tanh(x)
            alpha = self.param(f"alpha_{idx}", constant(self.nonlinearity), (1,))
            x = alpha * x + (1 - alpha) * identity

        return Dense(x.shape[-1], self.out_dim)(x)
    

class PIKAN(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True
    emb_scale: tuple = (2.0, 2.0)
    emb_dim: int = 64
    n_heads: int = 4

    def setup(self):
        self.act_fn = get_activation(self.act_name)
        # n_in = self.emb_dim * 4 if self.fourier_emb else 2
        # self.kan = KAN(
        #     layer_dims=[n_in] + [self.hidden_dim] * self.num_layers + [self.out_dim],
        #     layer_type="base",
        #     required_parameters={},
        # )

    @nn.compact
    def __call__(self, x, t):
        if self.fourier_emb:
            if len(self.emb_scale) == 1:
                x = jnp.concatenate([x, t], axis=-1)
                x = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
            else:
                t_emb = FourierEmbedding(self.emb_scale[1], self.emb_dim)(t)
                x_emb = FourierEmbedding(self.emb_scale[0], self.emb_dim)(x)
                x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        # 动态创建KAN，避免维度不匹配
        if not hasattr(self, '_kan_initialized'):
            actual_input_dim = x.shape[-1]
            self.kan = KAN(
                layer_dims=[actual_input_dim] + [self.hidden_dim] * self.num_layers + [self.out_dim],
                layer_type="base",
                required_parameters={},
            )
            self._kan_initialized = True
        
        x = self.kan(x)
        return x

        

