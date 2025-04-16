import datetime
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
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
            self.loss_bc,
            self.loss_irr,
        ]
        self.flux_idx = 1

    def ref_sol_bc_top(self, x, t):
        # uy = 0.007 * 0.78 / np.tanh(3) * np.tanh(3 * t)
        uy = self.cfg.loading(t[0])
        return jax.lax.stop_gradient(jnp.array([0.0, 0.0, uy]))

    def ref_sol_bc_bottom(self, x, t):
        return jax.lax.stop_gradient(jnp.array([0.0, 0.0, 0.0]))

    def ref_sol_bc_right(self, x, t):
        return jax.lax.stop_gradient(0.0)

    def ref_sol_bc_crack(self, x, t):
        # phi = exp(-|y| / l)
        phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L))
        return jax.lax.stop_gradient(phi)

    def ref_sol_ic(self, x, t):
        # if y > 0, phi = 0
        # elif y < 0, phi = exp(-|y| / l)
        # ux = 0, uy = 0
        phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L)) * jnp.where(x[0] < 0, 1, 0)
        sol = jnp.stack([phi, 0.0, 0.0], axis=-1)
        return jax.lax.stop_gradient(sol)

    def loss_ic(self, params, batch):
        x, t = batch
        phi, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        phi = phi[:, 0]
        # sol = jnp.stack([phi[:, 0], disp[:, 0], disp[:, 1]], axis=-1)
        ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
        return jnp.mean((phi - ref[:, 0]) ** 2) + jnp.mean(disp**2) * 1e4

    def loss_bc(self, params, batch):

        x, t = batch["bottom"]
        phi, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        # sol = jnp.stack([phi[:, 0], disp[:, 0], disp[:, 1]], axis=-1)
        ref = vmap(self.ref_sol_bc_bottom, in_axes=(0, 0))(x, t)
        phi = phi[:, 0]
        ux = disp[:, 0]
        uy = disp[:, 1]
        # bottom = jnp.mean((sol - ref) ** 2)
        bottom = (
            jnp.mean((phi - ref[:, 0]) ** 2)
            + jnp.mean((ux - ref[:, 1]) ** 2) * 1e4
            + jnp.mean((uy - ref[:, 2]) ** 2) * 1e4
        )

        x, t = batch["top"]
        phi, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        # ux = disp[:, 0]
        uy = disp[:, 1]
        phi = phi[:, 0]
        ref = vmap(self.ref_sol_bc_top, in_axes=(0, 0))(x, t)
        # ux should be constant using poisson coefficient `nu`
        # epsilon = vmap(self.epsilon, in_axes=(None, 0, 0))(
        #     params, x, t
        # )
        # eps_xx = epsilon[:, 0, 0]
        # eps_yy = epsilon[:, 1, 1]
        top = (
            jnp.mean((phi - ref[:, 0]) ** 2)
            + jnp.mean((uy - ref[:, 2]) ** 2) * 1e4
            # + jnp.mean((eps_xx + self.cfg.NU * eps_yy) ** 2) * 1e4
        )

        x, t = batch["crack"]
        phi = vmap(
            lambda x, t: self.net_u(params, x, t)[0],
            in_axes=(0, 0),
        )(
            x, t
        )[:, 0]
        ref = vmap(self.ref_sol_bc_crack, in_axes=(0, 0))(x, t)
        crack = jnp.mean((phi - ref) ** 2)

        return bottom + 10*top + crack

    # def loss_poison(self, params, batch):
    #     x, t = batch
    #     epsilon = vmap(self.epsilon, in_axes=(None, 0, 0))(
    #         params, x, t
    #     )


causal_first_point = 0.3
causal_bins_tail = jnp.linspace(
    causal_first_point, cfg.DOMAIN[-1][-1], cfg.CAUSAL_CONFIGS["chunks"]
)
# insert 0.0 at the beginning of the bins
causal_bins = jnp.insert(causal_bins_tail, 0, 0.0)
causal_weightor = CausalWeightor(
    cfg.CAUSAL_CONFIGS["chunks"], t_range=cfg.DOMAIN[-1], bins=causal_bins
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
    xdim=len(cfg.DOMAIN) - 1,
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

stagger = StaggerSwitch(pde_names=["stress","pf"], stagger_period=cfg.STAGGER_PERIOD)

start_time = time.time()
for epoch in range(cfg.EPOCHS):
    pde_name = stagger.decide_pde()
    loss_fn = pinn.loss_fn_stress if pde_name == "stress" else pinn.loss_fn_pf

    if epoch % cfg.STAGGER_PERIOD == 0:
        batch = sampler.sample(fns=[pinn.net_stress, pinn.net_pf], params=state.params)

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

        print(
            f"Epoch: {epoch}, "
            f"Error_phi: {error[0]:.2e}, "
            f"Error_ux: {error[1]:.2e}, "
            f"Error_uy: {error[2]:.2e}, "
            f"Loss_{pde_name}: {loss_components[0]:.2e}, "
        )

        metrics_tracker.register_scalars(
            epoch,
            names=[
                "loss/weighted",
                f"loss/{pde_name}",
                "loss/ic",
                "loss/bc",
                "loss/irr",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc",
                "weight/irr",
                "error/error_phi",
                "error/error_ux",
                "error/error_uy",
            ],
            values=[weighted_loss, *loss_components, *weight_components, *error],
        )
        metrics_tracker.register_figure(epoch, fig, "error")
        plt.close(fig)

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
