"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""

import datetime
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
from jax import jit, random, vmap

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from pinn import *
from examples.ice_melting.configs import Config as cfg


# from jax import config
# config.update("jax_disable_jit", True)


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

    def sample_pde_rar(self, pde_name="ac"):
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

    def sample(self, pde_name="ac"):
        return (
            self.sample_pde(),
            self.sample_ic(),
            # self.sample_pde(),
        )


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones(3), jnp.ones(1))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.adam(scheduler)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jit
def train_step(state, batch, eps):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(pinn.loss_fn, has_aux=True, argnums=0)(params, batch, eps)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)


class PFPINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            # self.loss_irr,
        ]

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        # r = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        # phi0 = jnp.tanh((self.cfg.R0 - r) / (jnp.sqrt(2) * self.cfg.EPSILON))
        phi = self.model.apply(params, x, t)
        return jax.nn.tanh(phi)

    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x, t):
        phi = self.net_u(params, x, t)
        dphi_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)
        hess_x = jax.hessian(self.net_u, argnums=1)(params, x, t)
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
        r = jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) * self.cfg.Lc
        phi = jnp.tanh((self.cfg.R0 - r) / (jnp.sqrt(2) * self.cfg.EPSILON))
        return jax.lax.stop_gradient(phi)
    
    @partial(jit, static_argnums=(0,))
    def net_speed(self, params, x, t):
        dphi_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)[0]
        return dphi_dt

    @partial(jit, static_argnums=(0,))
    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(jax.nn.relu(dphi_dt))
    


def evaluate3D(pinn, params, mesh, ref_path, ts, **kwargs):
    fig, axes = plt.subplots(
        len(ts),
        2,
        figsize=(10, 3 * len(ts)),
        subplot_kw={"projection": "3d", "box_aspect": (1, 1, 1)},
    )
    
    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim", (-0.5, 0.5))
    zlim = kwargs.get("zlim", (-0.5, 0.5))
    Lc = kwargs.get("Lc", 100)
    Tc = kwargs.get("Tc", 1.0)

    error = 0
    mesh /= Lc
    mesh = mesh[::10]
    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
        pred = vmap(lambda x, t: pinn.net_u(params, x, t)[0], in_axes=(0, 0))(
            mesh, t
        ).flatten()

        ax = axes[idx, 0]
        # interface_idx = jnp.where((pred > 0.05) & (pred < 0.95))[0]
        interface_idx = jnp.where((pred > -0.5) & (pred < 0.5))[0]
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
        # reverse z axis
        ax.invert_zaxis()
        
        ref_sol = jnp.load(f"{ref_path}/sol-{tic:.4f}.npy")[::10]
        diff = jnp.abs(pred - ref_sol)
        interface_idx = jnp.where((diff > 0.1))[0]
        ax = axes[idx, 1]
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
            title=f"t={tic}",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        # colorbar for error
        plt.colorbar(error_bar, ax=ax)
        error += jnp.mean(diff ** 2)
        
        ax.set_axis_off()
        ax.invert_zaxis()
        
    plt.tight_layout()
    error /= len(ts)
    return fig, error



pinn = PFPINN(config=cfg)


init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model, model_key, cfg.LR, decay=cfg.DECAY, decay_every=cfg.DECAY_EVERY
)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
sampler = Sampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "model": pinn,
        "params": state.params,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)
stagger = StaggerSwitch(pde_names=["ac"], stagger_period=cfg.STAGGER_PERIOD)
error = 0
start_time = time.time()
for epoch in range(cfg.EPOCHS):
    pde_name = stagger.decide_pde()
    pinn.pde_name = pde_name
    pinn.causal_weightor.pde_name = pde_name

    if epoch % cfg.STAGGER_PERIOD == 0:
        sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample(pde_name=pde_name)
        print(f"Epoch: {epoch}, PDE: {pde_name}")

    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        state, batch, cfg.CAUSAL_CONFIGS["eps"]
    )
    if cfg.CAUSAL_WEIGHT:
        update_causal_eps(aux_vars["causal_weights"], cfg.CAUSAL_CONFIGS, pde_name)
    stagger.step_epoch()

    if epoch % cfg.STAGGER_PERIOD == 0:

        # save the model
        params = state.params
        model_path = f"{log_path}/model-{epoch}.npz"
        params = jax.device_get(params)
        jnp.savez(model_path, **params)

        # if epoch > 100:
        fig, error = evaluate3D(
            pinn,
            state.params,
            jnp.load(f"{cfg.DATA_PATH}/mesh_points.npy"),
            cfg.DATA_PATH,
            ts=cfg.TS,
            Lc=cfg.Lc,
            Tc=cfg.Tc,
            xlim=cfg.DOMAIN[0],
            ylim=cfg.DOMAIN[1],
            zlim=cfg.DOMAIN[2],
        )

        print(
            f"Epoch: {epoch}, "
            f"Error: {error:.2e}, "
            f"Loss_{pde_name}: {loss_components[0]:.2e}, "
        )

        metrics_tracker.register_scalars(
            epoch,
            names=[
                "loss/weighted",
                f"loss/{pde_name}",
                "loss/ic",
                # "loss/bc",
                # "loss/irr",
                f"weight/{pde_name}",
                "weight/ic",
                # "weight/bc",
                # "weight/irr",
                "error/error",
            ],
            values=[weighted_loss, *loss_components, *weight_components, error],
        )
        metrics_tracker.register_figure(epoch, fig, "error")
        plt.close(fig)

        if cfg.CAUSAL_WEIGHT:
            fig = pinn.causal_weightor.plot_causal_info(
                pde_name,
                aux_vars["causal_weights"],
                aux_vars["loss_chunks"],
                cfg.CAUSAL_CONFIGS[pde_name + "_eps"],
            )
            metrics_tracker.register_figure(epoch, fig, "causal_info")
            plt.close(fig)

        metrics_tracker.flush()


end_time = time.time()
print(f"Training time: {end_time - start_time}")
