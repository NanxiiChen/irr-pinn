"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""

import datetime
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
from jax import jit, random
import orbax.checkpoint as ocp

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from pinn import MetricsTracker, CausalWeightor, update_causal_eps
from examples.ice_melting_sphere.configs import Config as cfg
from examples.ice_melting_sphere.model import PINN, Sampler, evaluate3D


# from jax import config
# config.update("jax_disable_jit", True)


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones(3), jnp.ones(1))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(scheduler),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


causal_weightor = CausalWeightor(cfg.CAUSAL_CONFIGS["chunks"], cfg.DOMAIN[-1])
pinn = PINN(config=cfg, causal_weightor=causal_weightor)
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


@jit
def train_step(
    state,
    batch,
):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux)), grads = (
        jax.value_and_grad(pinn.loss_fn, has_aux=True, argnums=0)(params, batch)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux)


error = 0
start_time = time.time()
for epoch in range(cfg.EPOCHS):

    if epoch % cfg.STAGGER_PERIOD == 0:
        sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample()

    state, (weighted_loss, loss_components, weight_components, aux) = train_step(
        state,
        batch,
    )
    if cfg.CAUSAL_WEIGHT:
        new_causal_configs = update_causal_eps(
            aux["causal_weights"], cfg.CAUSAL_CONFIGS
        )
        cfg.CAUSAL_CONFIGS.update(new_causal_configs)

    if epoch % cfg.STAGGER_PERIOD == 0:

        # save the model
        # params = state.params
        # model_path = f"{log_path}/model-{epoch}.npz"
        # params = jax.device_get(params)
        # jnp.savez(model_path, **params)
        ckpt.save(log_path + f"/model-{epoch}", state)

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
