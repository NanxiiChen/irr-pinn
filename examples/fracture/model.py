from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, vmap

from .arch import (
    MLP,
    ModifiedMLP,
    ResNet,
    MixtureOfExperts,
    SpatialTemporalMLP,
    SplitModifiedMLP
)
from pinn import CausalWeightor


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
        causal_weightor: CausalWeightor = None,
        **kwargs,
    ):
        super().__init__()

        loss_terms = kwargs.get("loss_terms", [])
        self.loss_fn_panel = [
            getattr(self, f"loss_{term}") for term in loss_terms]
        self.cfg = config
        arch = {
            "mlp": MLP,
            "modified_mlp": ModifiedMLP,
            "resnet": ResNet,
            "moe": MixtureOfExperts,
            "spatial_temporal_mlp": SpatialTemporalMLP,
            "split_modified_mlp": SplitModifiedMLP,
        }
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
        self.loss_fn_stress_x = partial(self.loss_fn, pde_name="stress_x")
        self.loss_fn_stress_y = partial(self.loss_fn, pde_name="stress_y")
        self.loss_fn_pf = partial(self.loss_fn, pde_name="pf")
        self.loss_fn_energy = partial(self.loss_fn, pde_name="energy")

    @partial(jit, static_argnums=(0,))
    def scale_phi(self, phi, beta=1e-3):
        # if abs(phi)<=2 return phi/4+1/2
        # elif phi<-2 return beta*(phi+2)
        # elif phi >2 return beta*(phi-2)+1
        return jnp.where(
            jnp.abs(phi) <= 2,
            phi / 4 + 1 / 2,
            jnp.where(
                phi < -2,
                beta * (phi + 2),
                beta * (phi - 2) + 1,
            ),
        )

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        sol = self.model.apply(params, x, t)
        phi, disp = jnp.split(sol, [1], axis=-1)
        # scale_factor = jnp.array([1.0, 0.5]) * self.cfg.DISP_PRE_SCALE
        # disp = disp / scale_factor
        # phi = jnp.tanh(phi) / 2 + 0.5
        phi = jax.nn.sigmoid(phi)
        # phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L)) * jnp.exp(-phi**2)
        # phi = self.scale_phi(phi)
        # phi = jnp.exp(-phi**2)
        # phi = jnp.exp(-jnp.abs(phi))
        # phi = jnp.exp(-jnp.abs(phi)*10)
        # phi = jnp.exp(-jax.nn.sigmoid(-phi*10)*10)
        # phi = jnp.exp(-jax.nn.softplus(-phi))

        # phi = jnp.exp(-jnp.abs(x[1]) / self.cfg.L) * (jnp.tanh(phi) / 2 + 0.5)

        # apply hard constraint on displacement
        y0, y1 = self.cfg.DOMAIN[1]
        ux, uy = jnp.split(disp, 2, axis=-1)
        ux = (x[1]-y0)*(x[1] - y1)*ux*self.cfg.loading(t)
        uy = (x[1]-y0)*(x[1] - y1)*uy*self.cfg.loading(t) + \
            (x[1]-y0)/(y1-y0)*self.cfg.loading(t)
        # # uy = (y1 - x[1])/ (y1-y0) * uy + (x[1]-y0) / (y1-y0) * self.cfg.loading(t)
        disp = jnp.concatenate((ux, uy), axis=-1)
        return phi, disp

    @partial(jit, static_argnums=(0,))
    def epsilon(self, params, x, t):
        # epsilon: (d, d)
        # $$ \varepsilon = sym(\nabla u) = \frac{1}{2}(\nabla u + (\nabla u)^T) $$
        nabla_disp = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )
        return (nabla_disp + jnp.transpose(nabla_disp, axes=(1, 0))) / 2.0

    @partial(jit, static_argnums=(0,))
    def psi(self, params, x, t):
        epsilon = self.epsilon(params, x, t)
        return self.cfg.LAMBDA * jnp.linalg.trace(epsilon) ** 2 / 2 \
            + self.cfg.MU * jnp.linalg.trace(epsilon @ epsilon)

    @partial(jit, static_argnums=(0,))
    def psi_pos(self, epsilon):
        tr_eps = jnp.trace(epsilon)
        k = self.cfg.LAMBDA + self.cfg.MU * 2 / self.cfg.DIM
        dev_eps = epsilon - tr_eps * jnp.eye(self.cfg.DIM) / self.cfg.DIM
        tr_deveps2 = jnp.linalg.trace(dev_eps @ dev_eps)
        return k * jax.nn.relu(tr_eps) ** 2 / 2 + self.cfg.MU * tr_deveps2

    @partial(jit, static_argnums=(0,))
    def psi_neg(self, epsilon):
        tr_eps = jnp.trace(epsilon)
        k = self.cfg.LAMBDA + self.cfg.MU * 2 / self.cfg.DIM
        return k * jax.nn.relu(-tr_eps) ** 2 / 2

    @partial(jit, static_argnums=(0,))
    def sigma(self, params, x, t):
        phi, disp = self.net_u(params, x, t)
        phi = phi.squeeze(-1)
        g_phi = (1 - phi) ** 2 + 1e-6
        epsilon = self.epsilon(params, x, t)
        dpsipos_deps = jax.grad(self.psi_pos, argnums=0)(epsilon)
        dpsineg_deps = jax.grad(self.psi_neg, argnums=0)(epsilon)
        return g_phi * dpsipos_deps + dpsineg_deps
        # sigma: (d, d)
        # return 2.0 * self.cfg.MU * self.epsilon(
        #     params, x, t
        # ) + self.cfg.LAMBDA * jnp.trace(self.epsilon(params, x, t)) * jnp.eye(
        #     self.cfg.DIM
        # )

    # @partial(jit, static_argnums=(0,))
    # def net_stress(self, params, x, t):
    #     # $$ \nabla\cdot[(1-\phi)^2 \sigma] = (1-\phi)^2 \nabla\cdot\sigma - 2(1-\phi)\sigma\cdot\nabla\phi = 0 $$
    #     phi, _ = self.net_u(params, x, t)
    #     # sigma[i,j]: sigma_ij
    #     sigma = self.sigma(params, x, t)
    #     # nabla_phi[i]: dphi / dx_i
    #     nabla_phi = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
    #         x, t
    #     )[0]
    #     # jac_sigma[i,j,k]: dsigma_ij / dx_k
    #     jac_sigma_x = jax.jacrev(self.sigma, argnums=1)(params, x, t)
    #     # div_sigma[i]: dsigma_ij / dx_i
    #     div_sigma = jnp.einsum("ijj->i", jac_sigma_x)
    #     # sigma_cdot_nabla_phi[i]: sigma_ij * dphi / dx_j
    #     sigma_cdot_nabla_phi = jnp.einsum("ij,j->i", sigma, nabla_phi)
    #     stress = (1 - phi) ** 2 * div_sigma - 2 * (1 - phi) * sigma_cdot_nabla_phi

    #     # stress = jnp.sum(jnp.abs(stress), axis=-1)
    #     # stress = jnp.sqrt(jnp.sum(stress**2, axis=-1))

    #     return stress / self.cfg.STRESS_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_stress(self, params, x, t):
        def damaged_sigma(x, t):
            phi, _ = self.net_u(params, x, t)
            sigma = self.sigma(params, x, t)
            return (1 - phi) ** 2 * sigma

        div_damaged_sigma = jax.jacrev(damaged_sigma, argnums=0)(x, t)
        stress = jnp.einsum("ijj->i", div_damaged_sigma)
        return stress / self.cfg.STRESS_PRE_SCALE
        # return jnp.linalg.norm(stress) / self.cfg.STRESS_PRE_SCALE

    def net_stress_x(self, params, x, t):
        return self.net_stress(params, x, t)[0]

    def net_stress_y(self, params, x, t):
        return self.net_stress(params, x, t)[1]

    @partial(jit, static_argnums=(0,))
    def _net_pf(self, params, x, t):
        phi, disp = self.net_u(params, x, t)
        epsilon = self.epsilon(params, x, t)
        phi = phi.squeeze(-1)
        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )[0]
        lap_phi = jnp.linalg.trace(hess_phi_x)

        pf = self.cfg.GC * (phi / self.cfg.L - self.cfg.L * lap_phi) - 2 * (
            1 - phi
        ) * self.psi_pos(epsilon)

        return pf / self.cfg.PF_PRE_SCALE

    def net_pf(self, params, x, t):
        pf = self._net_pf(params, x, t)
        dphi_dt = self.net_speed(params, x, t)
        phi, _ = self.net_u(params, x, t)
        # weights = 0 when dphi_dt = 0
        # weights = 0 when phi -> 1
        # weights = jax.lax.stop_gradient(
        #     jnp.where(dphi_dt <= 1e-3, 0.0, 1.0)
        # ) * jax.lax.stop_gradient(
        #     jnp.where(jnp.abs(phi-1) < 1e-3, 0.0, 1.0)
        # )
        # weights = jax.lax.stop_gradient(
        #     jnp.where((jnp.abs(dphi_dt) <= 1e-3) | (jnp.abs(phi-1) < 1e-3), 0.0, 1.0)
        # )
        # pf = weights * pf

        # # pf > 0 for dphi_dt = 0 and phi < 1, indicates not yet reached critical state
        mask_pos_pf = (jnp.abs(dphi_dt) <= 1e-3) & (jnp.abs(phi-1) > 1e-3)
        # # pf < 0 for phi = 1, indicates already totally fractured
        mask_neg_pf = jnp.abs(phi-1) <= 1e-3
        # # pf = 0 for dphi_dt > 0, indicates just at the critical state, cracking is growing
        pf = jnp.where(
            mask_pos_pf,
            jax.nn.relu(-pf),
            jnp.where(
                mask_neg_pf,
                jax.nn.relu(pf),
                pf
            )
        )
        return pf

    # def net_pf(self, params, x, t):
    #     pf = self._net_pf(params, x, t)
    #     dphi_dt = self.net_speed(params, x, t)
    #     # # Apply KKT conditions
    #     # pf = jnp.where(
    #     #     jnp.abs(dphi_dt) > 1e-3,  # dphi_dt != 0, cracking is growing,
    #     #     pf,                      # indicating critical state, pf = 0
    #     #     0,                    # dphi_dt = 0, pf can be any value
    #     # )
    #     # return pf

    #     return jnp.array([
    #         jax.nn.relu(-pf), # pf >=0
    #         dphi_dt*pf,
    #     ])

    @partial(jit, static_argnums=(0,))
    def complementarity(self, params, x, t):
        # complementarity condition: $d\phi/dt \cdot pf = 0$
        pf = self._net_pf(params, x, t)
        dphi_dt = self.net_speed(params, x, t)
        # return dphi_dt * pf # 这种容易形成始终 dphi_dt = 0 的情况，因为 dphi_dt =0 训练起来更简单
        weights = jax.lax.stop_gradient(
            # 认为 pf <= 1.0 时，是接近临界开裂状态，对应的 dphi_dt 可以是任意值
            jnp.where(pf <= self.cfg.PF_EPS, 0.0, pf)
        )
        return weights * dphi_dt

    # @partial(jit, static_argnums=(0,))
    # def net_energy(self, params, x, t):
    #     # use energy type formulation
    #     # $$(1-\phi)^2 \psi(\varepsilon) + G_c ( \frac{1}{2l}\phi^2 + \frac{l}{2} |\nabla\phi|^2 )$$
    #     phi, disp = self.net_u(params, x, t)
    #     phi = phi.squeeze(-1)

    #     nabla_phi = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
    #         x, t
    #     )[0]
    #     pf = (
    #         (1 - phi) ** 2 * self.psi_pos(params, x, t)
    #         + self.psi_neg(params, x, t)
    #         + self.cfg.GC
    #         * (
    #             phi**2 / (2 * self.cfg.L)
    #             + (self.cfg.L / 2) * jnp.sum(nabla_phi**2, axis=-1)
    #         )
    #     )
    #     return pf / self.cfg.PF_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_energy(self, params, x, t):
        phi, _ = self.net_u(params, x, t)
        eps = self.epsilon(params, x, t)
        phi = phi.squeeze(-1)
        nabla_phi = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )[0]
        return (
            (1-phi)*2 * self.psi_pos(eps) + self.psi_neg(eps)
            + self.cfg.GC * (
                phi**2 / (2 * self.cfg.L)
                + (self.cfg.L / 2) * jnp.sum(nabla_phi**2, axis=-1)
            )
        )

    def net_speed(self, params, x, t):
        # jac_dt = jax.jacrev(self.net_u, argnums=2)
        # dphi_dt, dc_dt = jac_dt(params, x, t)
        dphi_dt = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=1)(x, t)[
            0, 0
        ]
        return dphi_dt

    def ref_sol_ic(self, x, t):
        raise NotImplementedError

    def ref_sol_bc(self, x, t):
        raise NotImplementedError

    @partial(jit, static_argnums=(0, 4))
    def loss_pde(self, params, batch, eps, pde_name: str):
        x, t = batch

        fn = getattr(self, f"net_{pde_name}")
        residual = vmap(fn, in_axes=(None, 0, 0))(params, x, t)
        if pde_name == "stress" or pde_name == "pf":
            mse_res = jnp.mean(residual**2, axis=0)
            weights = jax.lax.stop_gradient(
                jnp.mean(mse_res, axis=-1) / (mse_res + 1e-6)
            )
            # repeat weights to match the length of residual, [batch_size, 2]
            weights = weights[None, :]
            residual = jnp.sqrt(jnp.sum(residual**2 * weights, axis=-1))

        # point-wise weight
        if self.cfg.POINT_WISE_WEIGHT:
            nabla_phi_fn = jax.jacrev(
                lambda params, x, t: self.net_u(params, x, t)[0], argnums=1
            )
            nabla_phi = vmap(
                lambda params, x, t: nabla_phi_fn(params, x, t)[0], in_axes=(None, 0, 0)
            )(params, x, t)
            grad_phi = jnp.sum(nabla_phi**2, axis=-1)
            # grad_phi = jnp.linalg.norm(nabla_phi, ord=2, axis=-1)
            weights = jax.lax.stop_gradient(1 / (1.0 + grad_phi))
            # weights = jax.lax.stop_gradient(jnp.exp(-grad_phi))
            residual = weights * residual
        else:
            weights = jax.lax.stop_gradient(jnp.ones_like(residual))

        if self.cfg.DEAD_POINTS_WEIGHT:
            phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
            phi = phi.squeeze(-1)
            weights = jax.lax.stop_gradient(
                jnp.where(jnp.abs(phi-1) < 1e-3, 0.0, 1.0)
            )
            residual = weights * residual
        else:
            weights = jax.lax.stop_gradient(jnp.ones_like(residual))

        if not self.cfg.CAUSAL_WEIGHT:
            if pde_name == "energy":
                return jnp.log(jnp.sum(residual)), {"weights": weights}
            return jnp.mean(residual**2), {"weights": weights}
        else:
            # loss, aux_vars = self.causal_weightor.compute_causal_loss(residual, t, eps)
            phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
            phi = jax.lax.stop_gradient(phi)
            causal_data = jnp.stack((t.reshape(-1),), axis=0)
            # causal_data = jnp.stack((t.reshape(-1),), axis=0)
            loss, aux_vars = self.causal_weightor.compute_causal_loss(
                residual,
                causal_data,
                eps
            )
            aux_vars.update({"weights": weights})
            return loss, aux_vars

    def loss_ic(self, params, batch):
        raise NotImplementedError

    def loss_bc(self, params, batch):
        raise NotImplementedError

    def loss_irr(self, params, batch, *args, **kwargs):
        x, t = batch
        dphi_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(nn.relu(-dphi_dt)), {
            "x_irr": x,
            "t_irr": t,
            "dphi_dt": dphi_dt,
        }

    def loss_irr_pf(self, params, batch, *args, **kwargs):
        x, t = batch
        pf = vmap(self._net_pf, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(nn.relu(-pf)), {}

    def loss_complementarity(self, params, batch, *args, **kwargs):
        x, t = batch
        comp = vmap(self.complementarity, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(comp**2), {}

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

            (loss_item, aux), grad_item = jax.value_and_grad(
                loss_item_fn, has_aux=True
            )(params, batch_item, eps, pde_name)
            aux_vars.update(aux)
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
        weights = weights.at[-4].set(weights[-4] * 5)

        # weights = weights.at[1].set(weights[1] * 5)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-8):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2)
                              for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        # weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = 1.0 / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)

        return jax.lax.stop_gradient(weights)
