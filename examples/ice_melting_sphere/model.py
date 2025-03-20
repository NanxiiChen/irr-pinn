from examples.ice_melting_sphere.configs import Config as cfg
from pinn import *
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from jax import jit, random, vmap

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
        causal_weightor: CausalWeightor = None,
    ):
        super().__init__()

        self.cfg = config

        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
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

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        # phi0 = self.ref_sol_ic(x, t)
        # return phi0 + self.model.apply(params, x, t) * t
        return nn.tanh(self.model.apply(params, x, t))

    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x, t):
        phi = self.net_u(params, x, t)
        dphi_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)[0] / self.cfg.Tc
        # note, if the `[0]` is not added, the shape of hess_x will be (1, 3, 3)
        # then the trace will be the sum of all the elements in the matrix
        # leading to totally wrong results, FUCK!!!
        hess_x = jax.hessian(self.net_u, argnums=1)(params, x, t)[0]
        lap_phi = jnp.linalg.trace(hess_x) / self.cfg.Lc**2

        Fphi = 0.25 * (phi**2 - 1) ** 2
        dFdphi = phi**3 - phi

        pde = (
            dphi_dt
            - self.cfg.MM * (lap_phi - dFdphi / self.cfg.EPSILON**2)
            + self.cfg.LAMBDA * jnp.sqrt(2 * Fphi) / self.cfg.EPSILON
        )
        return pde.squeeze()

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        r = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) * self.cfg.Lc
        phi = jnp.tanh((self.cfg.R0 - r) / (jnp.sqrt(2) * self.cfg.EPSILON))
        phi = jnp.expand_dims(phi, axis=-1)
        return jax.lax.stop_gradient(phi)

    @partial(jit, static_argnums=(0,))
    def net_speed(self, params, x, t):
        dphi_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)[0]
        return dphi_dt

    @partial(jit, static_argnums=(0,))
    def loss_ic(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
        return jnp.mean((u - ref) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(jax.nn.relu(dphi_dt))

    @partial(jit, static_argnums=(0,))
    def loss_pde(self, params, batch, eps):
        x, t = batch
        res = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(res**2), {}
        else:
            return self.causal_weightor.compute_causal_loss(
                res,
                t,
                eps,
            )

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch, eps):
        losses = []
        grads = []
        for idx, (loss_item_fn, batch_item) in enumerate(
            zip(self.loss_fn_panel, batch)
        ):
            if idx == 0:
                (loss_item, aux), grad_item = jax.value_and_grad(
                    loss_item_fn, has_aux=True, argnums=0
                )(params, batch_item, eps)
            else:
                loss_item, grad_item = jax.value_and_grad(loss_item_fn)(
                    params, batch_item
                )
            losses.append(loss_item)
            grads.append(grad_item)

        losses = jnp.array(losses)
        weights = self.grad_norm_weights(grads)
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)
        # weights = jax.lax.stop_gradient(jnp.array([1.0, 1.0, 1.0]))
        return jnp.sum(weights * losses), (losses, weights, aux)

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


def evaluate3D(pinn, params, mesh, ref_path, ts, **kwargs):
    fig = plt.figure(figsize=(3 * len(ts), 8))
    gs = GridSpec(
        4,
        len(ts) + 1,
        width_ratios=[0.1] + [1] * len(ts),
        height_ratios=[1, 1, 1, 0.3],
        figure=fig,
    )

    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim", (-0.5, 0.5))
    zlim = kwargs.get("zlim", (-0.5, 0.5))
    Lc = kwargs.get("Lc", 100)
    Tc = kwargs.get("Tc", 1.0)

    error = 0
    mesh /= Lc
    mesh = mesh[::10]

    row_names = ["PINN", "FEM", "Error"]
    for idx, row_name in enumerate(row_names):
        ax = fig.add_subplot(gs[idx, 0])
        # put row name on the left vertical axis
        ax.text(
            0.5,
            0.5,
            row_name,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
        pred = vmap(pinn.net_u, in_axes=(None, 0, 0))(params, mesh, t).squeeze()

        ax = fig.add_subplot(gs[0, idx + 1], projection="3d", box_aspect=(1, 1, 1))
        interface_idx = (pred > -0.5) & (pred < 0.5)
        ax.scatter(
            mesh[interface_idx, 0],
            mesh[interface_idx, 1],
            mesh[interface_idx, 2],
            c=pred[interface_idx],
            cmap="coolwarm",
            label="phi",
            vmin=-1,
            vmax=1,
        )
        r_pinn = (
            jnp.sqrt(
                mesh[interface_idx, 0] ** 2
                + mesh[interface_idx, 1] ** 2
                + mesh[interface_idx, 2] ** 2
            )
            * Lc
        )

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=f"t={tic}",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        ax.set_axis_off()
        ax.invert_zaxis()

        ref_sol = jnp.load(f"{ref_path}/sol-{tic:.4f}.npy")[::10]
        diff = jnp.abs(pred - ref_sol)

        ax = fig.add_subplot(gs[1, idx + 1], projection="3d", box_aspect=(1, 1, 1))
        interface_idx = (ref_sol > -0.5) & (ref_sol < 0.5)
        ax.scatter(
            mesh[interface_idx, 0],
            mesh[interface_idx, 1],
            mesh[interface_idx, 2],
            c=ref_sol[interface_idx],
            cmap="coolwarm",
            label="phi",
            vmin=-1,
            vmax=1,
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        ax.set_axis_off()

        ax = fig.add_subplot(gs[2, idx + 1], projection="3d", box_aspect=(1, 1, 1))
        interface_idx = diff > 0.05
        error_bar = ax.scatter(
            mesh[interface_idx, 0],
            mesh[interface_idx, 1],
            mesh[interface_idx, 2],
            c=jnp.abs(pred[interface_idx] - ref_sol[interface_idx]),
            cmap="coolwarm",
            label="error",
        )

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        error += jnp.mean(diff**2)

        ax.set_axis_off()
        ax.invert_zaxis()

        interface_idx = jnp.where((ref_sol > -0.5) & (ref_sol < 0.5))[0]
        r_fem = (
            jnp.sqrt(
                mesh[interface_idx, 0] ** 2
                + mesh[interface_idx, 1] ** 2
                + mesh[interface_idx, 2] ** 2
            )
            * Lc
        )
        r_analytical = cfg.R0 - cfg.LAMBDA * tic

        ax.text2D(
            0.05,
            -0.2,
            f"R_analytical = {r_analytical:.2f}\n"
            f"R_pinn = {jnp.mean(r_pinn):.2f}\n"
            f"R_fem = {jnp.mean(r_fem):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

        ax = fig.add_subplot(gs[3, idx + 1])
        fig.colorbar(error_bar, ax=ax, orientation="horizontal")
        ax.set_axis_off()

    error /= len(ts)
    return fig, error


class Sampler:

    def __init__(
        self,
        n_samples,
        domain=((-0.4, 0.4), (-0.4, 0.4), (0, 0.4), (0, 1)),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
            "model": None,
            "state": None,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def adaptive_sampling(self, residual_fn):
        key, self.key = random.split(self.key)
        adaptive_base = lhs_sampling(
            self.mins,
            self.maxs,
            self.adaptive_kw["num"] * self.adaptive_kw["ratio"],
            key=key,
        )
        residuals = residual_fn(adaptive_base)
        max_residuals, indices = jax.lax.top_k(
            jnp.abs(residuals), self.adaptive_kw["num"]
        )
        return adaptive_base[indices]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples, self.n_samples * 2],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_pde_rar(self):
        key, self.key = random.split(self.key)
        batch = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples, self.n_samples * 2],
            key,
        )

        def residual_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["params"]
            x, t = batch[:, :-1], batch[:, -1:]
            return vmap(model.net_pde, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(residual_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[1][1], self.domain[2][1]],
            num=10000,
            key=key,
        )
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample(
        self,
    ):
        return (
            self.sample_pde_rar(),
            self.sample_ic(),
            self.sample_pde(),
        )
