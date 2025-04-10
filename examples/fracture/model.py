from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, vmap

from pinn import CausalWeightor, MLP, ModifiedMLP


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
        causal_weightor: CausalWeightor = None,
    ):
        super().__init__()

        self.cfg = config
        self.flux_idx = 1
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_irr,
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

        self.loss_fn_stress = partial(self.loss_fn, pde_name="stress")
        self.loss_fn_pf = partial(self.loss_fn, pde_name="pf")

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        sol = self.model.apply(params, x, t)
        phi, disp = jnp.split(sol, [1], axis=-1)
        disp = disp / self.cfg.DISP_PRE_SCALE
        phi = jnp.exp(-phi / 5)
        return phi, disp

    @partial(jit, static_argnums=(0,))
    def epsilon(self, params, x, t):
        # epsilon: (d, d)
        # $$ \varepsilon = sym(\nabla u) = \frac{1}{2}(\nabla u + (\nabla u)^T) $$
        nabla_disp = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )
        return (nabla_disp + nabla_disp.T) / 2

    @partial(jit, static_argnums=(0,))
    def sigma(self, params, x, t):
        # sigma: (d, d)
        return 2.0 * self.cfg.MU * self.epsilon(
            params, x, t
        ) + self.cfg.LAMBDA * jnp.trace(self.epsilon(params, x, t)) * jnp.eye(
            x.shape[-1]
        )

    @partial(jit, static_argnums=(0,))
    def psi(self, params, x, t):
        # psi: scalar
        epsilon = self.epsilon(params, x, t)
        tr_eps = jnp.trace(epsilon)
        pos_energy = (
            (1 / 2)
            * (self.cfg.LAMBDA + self.cfg.MU)
            * ((tr_eps + jnp.abs(tr_eps)) / 2) ** 2
        )
        dev_eps = epsilon - tr_eps * jnp.eye(x.shape[-1]) / x.shape[-1]
        # Frobenius norm
        l2_eps = jnp.linalg.norm(dev_eps, ord="fro") ** 2
        return pos_energy + self.cfg.MU * l2_eps
        # return self.cfg.LAMBDA * jnp.trace(epsilon) ** 2 / 2 + self.cfg.MU * jnp.sum(
        #     epsilon**2
        # )

    @partial(jit, static_argnums=(0,))
    def net_stress(self, params, x, t):
        # $$ (1 - \phi) **2 \nabla \cdot \sigma = 0 $$
        phi, _ = self.net_u(params, x, t)
        jac_sigma_x = jax.jacrev(self.sigma, argnums=1)(params, x, t)

        # sigma[i,j]: sigma_ij
        # jac_sigma[i,j,k]: dsigma_ij / dx_k
        # div_sigma[i]: dsigma_ij / dx_i
        div_sigma = jnp.einsum("iji->j", jac_sigma_x)

        stress = (1 - phi) ** 2 * div_sigma
        weights = jax.lax.stop_gradient(
            jnp.sum(jnp.abs(stress), axis=-1) / (jnp.abs(stress) + 1e-6)
        )
        weighted_stress = jnp.sum(stress * weights, axis=-1)

        # stress = jnp.sum(jnp.abs(stress), axis=-1)
        return weighted_stress / self.cfg.STRESS_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_pf(self, params, x, t):
        phi, disp = self.net_u(params, x, t)
        phi = phi.squeeze()
        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )[0]
        lap_phi = jnp.linalg.trace(hess_phi_x)

        pf = self.cfg.GC * (phi / self.cfg.L - self.cfg.L * lap_phi) - 2 * (
            1 - phi
        ) * self.psi(params, x, t)

        return pf / self.cfg.PF_PRE_SCALE

    def net_speed(self, params, x, t):
        # jac_dt = jax.jacrev(self.net_u, argnums=2)
        # dphi_dt, dc_dt = jac_dt(params, x, t)
        dphi_dt = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=1)(x, t)[
            0
        ]
        return dphi_dt

    def ref_sol_ic(self, x, t):
        raise NotImplementedError

    def ref_sol_bc(self, x, t):
        raise NotImplementedError

    # def net_nabla(
    #     self,
    #     params,
    #     x,
    #     t,
    # ):
    #     idx = self.flux_idx
    #     nabla_phi_part = jax.jacrev(
    #         lambda x, t: self.net_u(params, x, t)[0], argnums=0
    #     )(x, t)[idx]
    #     nabla_c_part = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
    #         x, t
    #     )[idx]
    #     return nabla_phi_part, nabla_c_part

    @partial(jit, static_argnums=(0, 4))
    def loss_pde(self, params, batch, eps, pde_name: str):
        x, t = batch
        residual = jax.lax.cond(
            pde_name == "stress",
            lambda operand: vmap(self.net_stress, in_axes=(None, 0, 0))(
                params, *operand
            ),
            lambda operand: vmap(self.net_pf, in_axes=(None, 0, 0))(params, *operand),
            operand=(x, t),
        )
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(residual**2), {}
        else:
            return self.causal_weightor.compute_causal_loss(residual, t, eps)

    def loss_ic(self, params, batch):
        raise NotImplementedError

    def loss_bc(self, params, batch):
        raise NotImplementedError

    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(jax.nn.relu(-dphi_dt))

    # def loss_flux(self, params, batch):
    #     x, t = batch
    #     dphi_dx, dc_dx = vmap(self.net_nabla, in_axes=(None, 0, 0))(params, x, t)
    #     return jnp.mean(dphi_dx**2) + jnp.mean(dc_dx**2)

    @partial(jit, static_argnums=(0, 4))
    def compute_losses_and_grads(self, params, batch, eps, pde_name):
        if len(batch) != len(self.loss_fn_panel):
            raise ValueError(
                "The number of loss functions "
                "should be equal to the number of items in the batch"
            )
        losses = []
        grads = []
        aux_vars = {}
        for idx, (loss_item_fn, batch_item) in enumerate(
            zip(self.loss_fn_panel, batch)
        ):
            if idx == 0:
                (loss_item, aux), grad_item = jax.value_and_grad(
                    loss_item_fn, has_aux=True
                )(params, batch_item, eps, pde_name)
                aux_vars.update(aux)
            else:
                loss_item, grad_item = jax.value_and_grad(loss_item_fn)(
                    params, batch_item
                )
            losses.append(loss_item)
            grads.append(grad_item)

        return jnp.array(losses), grads, aux_vars

    @partial(jit, static_argnums=(0, 4))
    def loss_fn(self, params, batch, eps, pde_name):
        losses, grads, aux_vars = self.compute_losses_and_grads(
            params, batch, eps, pde_name
        )

        weights = self.grad_norm_weights(grads)
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)

        # weights = weights.at[1].set(weights[1] * 10)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-6):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
