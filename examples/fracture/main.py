import datetime
import sys
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap, jit
import orbax.checkpoint as ocp

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from examples.fracture import (
    PINN,
    FractureSampler,
    evaluate2D,
    cfg,
)
from pinn import (
    CausalWeightor,
    MetricsTracker,
    train_step,
    create_train_state,
    StaggerSwitch,
)


# from jax import config
# config.update("jax_disable_jit", True)


class FracturePINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc_bottom_u,
            self.loss_bc_top_u,
            self.loss_bc_crack,
            self.loss_bc_right,
            # self.loss_bc_sigmax,
            self.loss_irr,
        ]
        self.flux_idx = 1

    def ref_sol_bc_top(self, x, t):
        # uy = 0.007 * 0.78 / np.tanh(3) * np.tanh(3 * t)
        uy = self.cfg.loading(t[0])
        return jax.lax.stop_gradient(jnp.array([0.0, 0.0, uy]))

    def ref_sol_bc_bottom(self, x, t):
        return jax.lax.stop_gradient(jnp.array([0.0, 0.0, 0.0]))

    def ref_sol_bc_crack(self, x, t):
        # phi = exp(-|y| / l)
        phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L))
        return jax.lax.stop_gradient(phi)

    def ref_sol_ic(self, x, t):
        phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L)) * (1 - jax.nn.sigmoid(x[0] * 200))
        return jax.lax.stop_gradient(phi)

    def loss_ic(self, params, batch):
        x, t = batch
        phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        phi = phi[:, 0]
        ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
        return jnp.mean((phi - ref) ** 2)

    def loss_bc_bottom_phi(self, params, batch):
        x, t = batch
        phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_bc_bottom, in_axes=(0, 0))(x, t)
        phi = phi[:, 0]
        bottom = jnp.mean((phi - ref[:, 0]) ** 2)
        return bottom

    def loss_bc_bottom_u(self, params, batch):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_bc_bottom, in_axes=(0, 0))(x, t)
        ux = disp[:, 0]
        uy = disp[:, 1]
        bottom = jnp.mean((ux - ref[:, 1]) ** 2) + jnp.mean((uy - ref[:, 2]) ** 2)
        return bottom

    def loss_bc_top_phi(self, params, batch):
        x, t = batch
        phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_bc_top, in_axes=(0, 0))(x, t)
        phi = phi[:, 0]
        top = jnp.mean((phi - ref[:, 2]) ** 2)
        return top

    def loss_bc_top_u(self, params, batch):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        # ux = disp[:, 0]
        uy = disp[:, 1]
        ref = vmap(self.ref_sol_bc_top, in_axes=(0, 0))(x, t)
        # ux should be constant using poisson coefficient `nu`
        # epsilon = vmap(self.epsilon, in_axes=(None, 0, 0))(
        #     params, x, t
        # )
        # eps_xx = epsilon[:, 0, 0]
        # eps_yy = epsilon[:, 1, 1]
        top = jnp.mean((uy - ref[:, 2]) ** 2)
        return top

    def loss_bc_crack(self, params, batch):
        x, t = batch
        phi = vmap(
            lambda x, t: self.net_u(params, x, t)[0],
            in_axes=(0, 0),
        )(
            x, t
        )[:, 0]
        ref = vmap(self.ref_sol_bc_crack, in_axes=(0, 0))(x, t)
        crack = jnp.mean((phi - ref) ** 2)

        return crack

    def loss_bc_right(self, params, batch):
        x, t = batch
        nabla_phi_fn = jax.jacrev(
            lambda params, x, t: self.net_u(params, x, t)[0], argnums=1
        )
        dphi_dx = vmap(
            lambda params, x, t: nabla_phi_fn(params, x, t)[0, 0], in_axes=(None, 0, 0)
        )(params, x, t)
        return jnp.mean(dphi_dx**2)

    def loss_bc_sigmax(self, params, batch):
        x, t = batch
        phi = vmap(
            lambda params, x, t: self.net_u(params, x, t)[0], in_axes=(None, 0, 0)
        )(params, x, t).squeeze(-1)
        sigmax = vmap(
            lambda params, x, t: self.sigma(params, x, t)[0, 0], in_axes=(None, 0, 0)
        )(params, x, t)
        res = (1 - phi) ** 2 * sigmax
        return jnp.mean(res**2)


causal_first_point = 0.3
causal_bins_tail = jnp.linspace(
    causal_first_point, cfg.DOMAIN[-1][-1], cfg.CAUSAL_CONFIGS["chunks"]
)
# insert 0.0 at the beginning of the bins
causal_bins = jnp.insert(causal_bins_tail, 0, 0.0)
causal_weightor = CausalWeightor(
    cfg.CAUSAL_CONFIGS["chunks"],
    t_range=cfg.DOMAIN[-1],
)
pinn = FracturePINN(config=cfg, causal_weightor=causal_weightor)

init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model,
    model_key,
    cfg.LR,
    decay=cfg.DECAY,
    decay_every=cfg.DECAY_EVERY,
    xdim=cfg.DIM,
    optimizer=cfg.OPTIMIZER,
)


if cfg.RESUME is not None:
    ckpt = ocp.StandardCheckpointer()
    restore_state = ckpt.restore(cfg.RESUME)
    # state load the params from the checkpoint
    state = state.replace(
        params=restore_state["params"],
    )
    print(f"Restored from {cfg.RESUME}")


now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
ckpt = ocp.StandardCheckpointer()
sampler = FractureSampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)

stagger = StaggerSwitch(pde_names=["stress", "pf"], stagger_period=cfg.STAGGER_PERIOD)

start_time = time.time()
for epoch in range(cfg.EPOCHS):

    if epoch == cfg.CHANGE_OPT_AT:
        print("Change optimizer to SOAP")
        current_params = state.params
        state = create_train_state(
            pinn.model,
            model_key,
            cfg.LR,
            xdim=cfg.DIM,
            optimizer="soap",
        )
        state = state.replace(params=current_params)

    pde_name = stagger.decide_pde()
    # loss_fn = pinn.loss_fn_stress if pde_name == "stress" else pinn.loss_fn_pf
    loss_fn = getattr(pinn, f"loss_fn_{pde_name}")

    if epoch % cfg.STAGGER_PERIOD == 0:
        batch = sampler.sample(
            fns=[pinn.psi, getattr(pinn, f"net_{pde_name}")],
            params=state.params,
        )

    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        loss_fn,
        state,
        batch,
        cfg.CAUSAL_CONFIGS[f"{pde_name}_eps"],
    )
    if cfg.CAUSAL_WEIGHT:
        new_eps = causal_weightor.update_causal_eps(
            cfg.CAUSAL_CONFIGS[f"{pde_name}_eps"],
            aux_vars["causal_weights"],
            cfg.CAUSAL_CONFIGS,
        )
        cfg.CAUSAL_CONFIGS.update({f"{pde_name}_eps": new_eps})

    stagger.step_epoch()

    if epoch % cfg.STAGGER_PERIOD == 0:

        # save the model
        if epoch % (10 * cfg.STAGGER_PERIOD) == 0:
            ckpt.save(log_path + f"/model-{epoch}", state)

            fig, error = evaluate2D(
                pinn,
                state.params,
                jnp.load(f"{cfg.DATA_PATH}/mesh_points.npy"),
                cfg.DATA_PATH,
                ts=cfg.TS,
                Lc=cfg.Lc,
                Tc=cfg.Tc,
                val_range=(0, 1),
                xlim=cfg.DOMAIN[0],
                ylim=cfg.DOMAIN[1],
            )
            metrics_tracker.register_figure(epoch, fig, "error")
            plt.close(fig)

        print(f"Epoch: {epoch}, " f"Loss_{pde_name}: {loss_components[0]:.2e}, ")

        metrics_tracker.register_scalars(
            epoch,
            names=[
                "loss/weighted",
                f"loss/{pde_name}",
                "loss/ic",
                "loss/bc_bottom_u",
                "loss/bc_top_u",
                "loss/bc_crack",
                "loss/bc_right",
                "loss/irr",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc_bottom_u",
                "weight/bc_top_u",
                "weight/bc_crack",
                "weight/bc_right",
                "weight/irr",
                # "error/error_phi",
                # "error/error_ux",
                # "error/error_uy",
            ],
            values=[
                weighted_loss,
                *loss_components,
                *weight_components,
            ],
        )

        if cfg.CAUSAL_WEIGHT:
            fig = pinn.causal_weightor.plot_causal_info(
                aux_vars["causal_weights"],
                aux_vars["loss_chunks"],
                cfg.CAUSAL_CONFIGS[f"{pde_name}_eps"],
            )
            fig.suptitle(f"{pde_name}_eps: {cfg.CAUSAL_CONFIGS[f'{pde_name}_eps']:.2e}")
            metrics_tracker.register_figure(epoch, fig, "causal_info")
            plt.close(fig)

        metrics_tracker.flush()


end_time = time.time()
print(f"Training time: {end_time - start_time}")
