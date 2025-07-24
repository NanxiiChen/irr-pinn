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
from jax import jit, random
import orbax.checkpoint as ocp

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from examples.corrosion2d1pit import (
    PINN,
    CorrosionSampler,
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


class CorrosionPINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_flux,
            self.loss_irr,
        ]
        self.flux_idx = 1

    @partial(jit, static_argnums=(0,))
    def ref_sol_bc(self, x, t):
        # x: (x1, x2)
        r = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
        phi = (r > 0.05).astype(jnp.float32)
        c = phi.copy()
        sol = jnp.stack([phi, c], axis=-1)
        return jax.lax.stop_gradient(sol)

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        r = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
        phi = (
            1
            - (
                1
                - jnp.tanh(
                    jnp.sqrt(cfg.OMEGA_PHI)
                    / jnp.sqrt(2 * cfg.ALPHA_PHI)
                    * (r - 0.05)
                    * cfg.Lc
                )
            )
            / 2
        )
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * cfg.CSE + (1 - h_phi) * 0.0
        sol = jnp.stack([phi, c], axis=-1)
        return jax.lax.stop_gradient(sol)


causal_weightor = CausalWeightor(cfg.CAUSAL_CONFIGS["chunks"], cfg.DOMAIN[-1])
pinn = CorrosionPINN(config=cfg, causal_weightor=causal_weightor)

init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model,
    model_key,
    cfg.LR,
    decay=cfg.DECAY,
    decay_every=cfg.DECAY_EVERY,
    xdim=len(cfg.DOMAIN) - 1,
    end_value=1e-5
)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
ckpt = ocp.StandardCheckpointer()
sampler = CorrosionSampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)

stagger = StaggerSwitch(pde_names=["ac", "ch"], stagger_period=cfg.STAGGER_PERIOD)

start_time = time.time()
for epoch in range(cfg.EPOCHS):
    pde_name = stagger.decide_pde()
    loss_fn = pinn.loss_fn_ac if pde_name == "ac" else pinn.loss_fn_ch

    if epoch % cfg.STAGGER_PERIOD == 0:
        batch = sampler.sample(fns=[pinn.net_ac, pinn.net_ch], params=state.params)

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
                "loss/bc",
                "loss/flux",
                "loss/irr",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc",
                "weight/flux",
                "weight/irr",
                "error/error",
            ],
            values=[weighted_loss, *loss_components, *weight_components, error],
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
