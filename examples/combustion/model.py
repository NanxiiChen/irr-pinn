from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, vmap

from .arch import MLP, ModifiedMLP, ResNet


class PINN(nn.Module):
    
    def __init__(
        self,
        config: object = None,
        causal_weightor = None,
    ):
        super().__init__()

        self.cfg = config

        self.loss_fn_panel = [
        ]
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[self.cfg.ARCH_NAME](
            act_name=self.cfg.ACT_NAME,
            num_layers=self.cfg.NUM_LAYERS,
            hidden_dim=self.cfg.HIDDEN_DIM,
            out_dim=self.cfg.OUT_DIM,
            fourier_emb=self.cfg.FOURIER_EMB,
            emb_scale=self.cfg.EMB_SCALE,
            emb_dim=self.cfg.EMB_DIM,
        )
        self.causal_weightor = causal_weightor


    @partial(jit, static_argnums=(0,))
    def net_T(self, params, x):
        return self.model.apply(params, x)
    
    @partial(jit, static_argnums=(0,))
    def net_sl(self, params, x):
        return self.model.sl
    
    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x):
        T = self.net_T(params, x)
        sl = self.net_sl(params, x)
        R = self.cfg.R
        T_IN = self.cfg.T_IN
        W = self.cfg.W
        c = sl + R * T_IN / (W * sl)
        return (c - jnp.sqrt(c**2 - 4*R*T / W)) / 2
    
    @partial(jit, static_argnums=(0,))
    def net_rho(self, params, x):
        u = self.velocity(params, x)
        RPO_IN = self.cfg.RHO_IN
        _, sl = self.net_u(params, x)
        return RPO_IN * sl / u
    
    @partial(jit, static_argnums=(0,))
    def net_yf(self, params, x):
        YF_IN = self.cfg.YF_IN
        QF = self.cfg.QF
        CP = self.cfg.CP
        T_IN = self.cfg.T_IN
        T = self.net_T(params, x)
        return YF_IN + CP * (T_IN - T) / QF
    
    @partial(jit, static_argnums=(0,))
    def net_omega(self, params, x):
        A = self.cfg.A
        EA = self.cfg.EA
        R = self.cfg.R
        NU = self.cfg.NU
        T = self.net_T(params, x)
        rho = self.net_rho(params, x)
        yf = self.net_yf(params, x)
        return A * jnp.exp(-EA / (R * T)) * (rho * yf) ** NU

    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x):
        RHO_IN = self.cfg.RHO_IN
        sl = self.net_sl(params, x)
        T = self.net_T(params, x)
        CP = self.cfg.CP
        LAMBDA = self.cfg.LAMBDA
        omega = self.net_omega(params, x)
        QF = self.cfg.QF
        dT_dx = jax.jacrev(self.net_T, argnums=1)(params, x)[0]
        d2T_dx2 = jax.hessian(self.net_T, argnums=1)(params, x)[0]
        pde = (
            RHO_IN * sl * CP * dT_dx
            - LAMBDA * d2T_dx2
            - omega * QF
        )

    @partial(jit, static_argnums=(0,))
    def loss_pde(self, params, batch,):
        x = batch
        res = vmap(self.net_pde, in_axes=(None, 0))(params, x)
        return jnp.mean(res**2), {}
    

    @partial(jit, static_argnums=(0,))
    def net_speed(self, params, x):
        dT_dx = jax.jacrev(self.net_T, argnums=1)(params, x)[0]
        return dT_dx

    @partial(jit, static_argnums=(0,))
    def loss_irr(self, params, batch):
        x = batch
        dT_dx = vmap(self.net_speed, in_axes=(None, 0))(params, x)
        return jnp.mean(jax.nn.relu(dT_dx)), {}
    

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch):
        losses = []
        grads = []
        aux_vars = {}
        for idx, (loss_item_fn, batch_item) in enumerate(
            zip(self.loss_fn_panel, batch)
        ):
            (loss_item, aux), grad_item = jax.value_and_grad(
                loss_item_fn, has_aux=True, argnums=0
            )(params, batch_item)
            losses.append(loss_item)
            grads.append(grad_item)
            aux_vars.update(aux)
        
        losses = jnp.array(losses)
        weights = self.grad_norm_weights(grads)
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)

        return jnp.sum(losses * weights), (losses, weights, aux_vars)
            
    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-8):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        # weights = 1.0 / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)

        return jax.lax.stop_gradient(weights)

    
    
    