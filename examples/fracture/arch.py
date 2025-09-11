from typing import Callable

import jax
from jax import vmap
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

        x = Dense(x.shape[-1], self.hidden_dim)(x)
        for _ in range(self.num_layers):
            x_res = x
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
            x = x + x_res

        phi = self.act_fn(Dense(x.shape[-1], self.hidden_dim)(x))
        phi = Dense(phi.shape[-1], 1)(phi)
        ux = self.act_fn(Dense(x.shape[-1], self.hidden_dim)(x))
        ux = Dense(ux.shape[-1], 1)(ux)
        uy = self.act_fn(Dense(x.shape[-1], self.hidden_dim)(x))
        uy = Dense(uy.shape[-1], 1)(uy)
        return jnp.concatenate([phi, ux, uy], axis=-1)
        # return Dense(self.hidden_dim, self.out_dim)(x)


class ModifiedMLPBlock(nn.Module):
    hidden_dim: int
    num_layers: int
    act_fn: callable
    out_dim: int = None  # Optional output dimension

    @nn.compact
    def __call__(self, x):
        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = nn.tanh(x)
            x = x * u + (1 - x) * v

        return Dense(x.shape[-1], self.out_dim or x.shape[-1])(x)

# class ModifiedMLP(nn.Module):
#     act_name: str = "tanh"
#     num_layers: int = 4
#     hidden_dim: int = 64
#     out_dim: int = 2
#     fourier_emb: bool = True
#     emb_scale: tuple = (2.0, 2.0)
#     emb_dim: int = 64

#     def setup(self):
#         self.act_fn = get_activation(self.act_name)

#         self.modified_mlp_block = ModifiedMLPBlock(
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

#         # x_features = self.modified_mlp_block(x_emb)
#         # t_features = self.modified_mlp_block(t_emb)
#         features = vmap(self.modified_mlp_block)(jnp.stack([x_emb, t_emb], axis=0))
#         combined = jnp.prod(features, axis=0)

#         # combined = (x_features + 1) * (t_features + 1) - 1

#         return self.output_layer(combined)


# class ModifiedMLP(nn.Module):
#     act_name: str = "tanh"
#     num_layers: int = 4
#     hidden_dim: int = 64
#     out_dim: int = 2
#     fourier_emb: bool = True
#     emb_scale: tuple = (2.0, 2.0)
#     emb_dim: int = 64

#     def setup(self):
#         self.act_fn = get_activation(self.act_name)

#         self.phi_block = MLPBlock(
#             hidden_dim=self.hidden_dim,
#             num_layers=self.num_layers,
#             act_fn=self.act_fn,
#             out_dim=1
#         )

#         self.disp_block = ModifiedMLPBlock(
#             hidden_dim=self.hidden_dim,
#             num_layers=self.num_layers,
#             act_fn=self.act_fn,
#             out_dim=self.out_dim - 1
#         )


        
#     @nn.compact
#     def __call__(self, x, t):

#         phi_features = jnp.concatenate([x, t], axis=-1)
#         disp_features = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(
#             jnp.concatenate([x, t], axis=-1)
#         )
#         phi = self.phi_block(phi_features)
#         disp = self.disp_block(disp_features)

#         return jnp.concatenate([phi, disp], axis=-1)




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

        x = jnp.concatenate([x, t], axis=-1)
        if self.fourier_emb:
            # x = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(x)
            x_emb = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(x)
            t_emb = FourierEmbedding(emb_scale=self.emb_scale[1], emb_dim=self.emb_dim)(t)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)

        # x_feat = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(x)
        # t_feat = Dense(t.shape[-1], self.emb_dim*2)(t)
        # x = jnp.concatenate([x_feat, t_feat], axis=-1)

        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            # x = self.act_fn(x)
            x = nn.tanh(x)
            x = x * u + (1 - x) * v

        x = self.act_fn(Dense(x.shape[-1], self.hidden_dim*3)(x))
        phi = MLPBlock(
            hidden_dim=self.hidden_dim,
            num_layers=2,
            act_fn=self.act_fn,
            out_dim=1
        )(x)
        ux = MLPBlock(
            hidden_dim=self.hidden_dim,
            num_layers=2,
            act_fn=self.act_fn,
            out_dim=1
        )(x)
        uy = MLPBlock(
            hidden_dim=self.hidden_dim,
            num_layers=2,
            act_fn=self.act_fn,
            out_dim=1
        )(x)
        return jnp.concatenate([phi, ux, uy], axis=-1)


class SpatialTemporalMLP(nn.Module):
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
            x_feat = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(x)
            # t_feat = FourierEmbedding(emb_scale=self.emb_scale[1], emb_dim=self.emb_dim)(t)
            t_feat = Dense(t.shape[-1], self.hidden_dim*2, name="t_dense_proj")(t)
        else:
            x_feat = x
            t_feat = t
            
        for i in range(self.num_layers):
            x_feat = Dense(x_feat.shape[-1], self.hidden_dim, name=f"x_dense_{i}")(x_feat)
            t_feat = Dense(t_feat.shape[-1], self.hidden_dim, name=f"t_dense_{i}")(t_feat)
            fused = x_feat * t_feat
            fused = self.act_fn(fused)

            x_feat = fused + x_feat
            t_feat = fused + t_feat

        return Dense(fused.shape[-1], self.out_dim, name="output_dense")(fused)




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
            x_emb = FourierEmbedding(emb_scale=self.emb_scale[0], emb_dim=self.emb_dim)(x)
            # t_emb = FourierEmbedding(emb_scale=self.emb_scale[1], emb_dim=self.emb_dim)(t)
            t_emb = self.act_fn(Dense(t.shape[-1], self.emb_dim*2)(t))
            x_proj = Dense(x_emb.shape[-1], self.hidden_dim)(x_emb)
            t_proj = Dense(t_emb.shape[-1], self.hidden_dim)(t_emb)
            x = x_proj * t_proj
        else:
            x = jnp.concatenate([x, t], axis=-1)

        gating_weights = GatingNetwork(
            n_experts=self.n_experts,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            act_name=self.act_name,
        )(x)

        expert_outputs = jnp.stack([expert(x) for expert in self.experts], axis=-1)

        output = jnp.sum(gating_weights[None, ...] * expert_outputs, axis=-1)
        return output
    

class SplitModifiedMLP(nn.Module):
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

        x = jnp.concatenate([x, t], axis=-1)
        phi = ModifiedMLPBlock(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            act_fn=self.act_fn,
            out_dim=1
        )(x)
        disp = ModifiedMLPBlock(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            act_fn=self.act_fn,
            out_dim=self.hidden_dim*2
        )(x)
        ux = Dense(disp.shape[-1], 1)(disp)
        uy = Dense(disp.shape[-1], 1)(disp)
        return jnp.concatenate([phi, ux, uy], axis=-1)

    