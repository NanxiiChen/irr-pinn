import datetime
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from jax import jit, random

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))



from examples.ice_melting import *
from pinn import CausalWeightor, MetricsTracker

# from jax import config
# config.update("jax_disable_jit", True)


class IceMeltingPINN(PINN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        r = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) * self.cfg.Lc
        phi = jnp.tanh((self.cfg.R0 - r) / (jnp.sqrt(2) * self.cfg.EPSILON))
        phi = jnp.expand_dims(phi, axis=-1)
        return jax.lax.stop_gradient(phi)



causal_weightor = CausalWeightor(cfg.CAUSAL_CONFIGS["chunks"], cfg.DOMAIN[-1])
pinn = IceMeltingPINN(config=cfg, causal_weightor=causal_weightor)


init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model, model_key, cfg.LR, decay=cfg.DECAY, decay_every=cfg.DECAY_EVERY
)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
ckpt = ocp.StandardCheckpointer()
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


error = 0
start_time = time.time()
for epoch in range(cfg.EPOCHS):

    if epoch % cfg.STAGGER_PERIOD == 0:
        sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample()

    state, (weighted_loss, loss_components, weight_components, aux) = (
        train_step(
            pinn.loss_fn,
            state,
            batch,
            cfg.CAUSAL_CONFIGS["eps"],
        )
    )
    if cfg.CAUSAL_WEIGHT:
        cfg.CAUSAL_CONFIGS.update(
            causal_weightor.update_causal_eps(aux["causal_weights"], cfg.CAUSAL_CONFIGS)
        )

    if epoch % cfg.STAGGER_PERIOD == 0:

        ckpt.save(log_path + f"/model-{epoch}", state)

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
            f"Loss_pde: {loss_components[0]:.2e}, "
        )

        metrics_tracker.register_scalars(
            epoch,
            names=[
                "loss/weighted",
                f"loss/pde",
                "loss/ic",
                "loss/irr",
                f"weight/pde",
                "weight/ic",
                "weight/irr",
                "error/error",
            ],
            values=[weighted_loss, *loss_components, *weight_components, error],
        )
        metrics_tracker.register_figure(epoch, fig, "error")
        plt.close(fig)

        if cfg.CAUSAL_WEIGHT:
            fig = causal_weightor.plot_causal_info(
                aux["causal_weights"],
                aux["loss_chunks"],
                cfg.CAUSAL_CONFIGS["eps"],
            )
            metrics_tracker.register_figure(epoch, fig, "causal")
            plt.close(fig)

        metrics_tracker.flush()


end_time = time.time()
print(f"Training time: {end_time - start_time}")
