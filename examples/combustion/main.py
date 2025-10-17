import datetime
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from jax import jit, random, vmap
jax.config.update("jax_enable_x64", True)  # 启用 float64 精度
# jax.config.update("jax_disable_jit", True)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from examples.combustion import (
    PINN,
    CombustionSampler,
    evaluate1D,
    cfg,
)
from examples.combustion.train import train_step, create_train_state

from pinn import (
    MetricsTracker,
)

class CombustionPINN(PINN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_bc_t(self, params, x):
        T = vmap(self.net_T, in_axes=(None, 0))(params, x)
        T_IN = self.cfg.T_IN
        return jnp.mean((T - T_IN) ** 2) / self.cfg.T_PRE_SCALE**2, {}
    
    def loss_bc_grad(self, params, x):
        dT_dx = vmap(self.net_speed, in_axes=(None, 0))(params, x)
        DTDX_IN = self.cfg.DTDX_IN
        return jnp.mean((dT_dx - DTDX_IN) ** 2) / self.cfg.T_PRE_SCALE**2, {}
    
    def loss_sl(self, params, x):
        # sl should be no less than 0
        sl = self.net_sl(params, x)
        minv = 0.1
        maxv = 1.0
        res = (sl - minv) * (maxv - sl)
        # res should be no less than 0
        return jnp.mean(jax.nn.relu(-res)), {}
    
    def loss_bc_u_in(self, params, x):
        u = vmap(self.net_u, in_axes=(None, 0))(params, x)
        sl = self.net_sl(params, x)
        return jnp.mean((u - sl) ** 2), {}
    
    def loss_bc_t_right(self, params, x):
        T = vmap(self.net_T, in_axes=(None, 0))(params, x)
        T_ADIA = self.cfg.T_ADIA
        return jnp.mean((T - T_ADIA) ** 2) / self.cfg.T_PRE_SCALE**2, {}

loss_terms = [
    "pde",
    "bc_t",
    "bc_grad",
    # "bc_t_right",
    # "sl",
    "irr",
]
pinn = CombustionPINN(config=cfg, loss_terms=loss_terms)

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
    end_value=1e-6,
    time_dependent=False
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
sampler = CombustionSampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)

start_time = time.time()
for epoch in range(cfg.EPOCHS):

    if epoch == cfg.CHANGE_OPT_AT:
        print(f"Changing optimizer to adam")
        current_params = state.params
        state = create_train_state(
            pinn.model,
            model_key,
            cfg.LR,
            decay=cfg.DECAY,
            decay_every=cfg.DECAY_EVERY,
            xdim=cfg.DIM,
            optimizer="adam",
            end_value=1e-6,
            time_dependent=False
        )
        state = state.replace(
            params=current_params,
        )




    if epoch % 10 == 0:
        batch = sampler.sample(
            fns=[pinn.net_pde,],
            params=state.params,
            rar=pinn.cfg.RAR,
            model=pinn,
        )

    # 交替冻结策略
    if epoch % 50 < 25:  # 前10个epoch冻结sl，只训练模型参数
        freeze_model = False
        freeze_sl = True
        training_phase = "model"
    else:  # 后10个epoch冻结模型，只训练sl参数
        freeze_model = True
        freeze_sl = False
        training_phase = "sl"

    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        pinn.loss_fn,
        state,
        batch,
        0.01,  # eps is not used in this case
        freeze_model=freeze_model,
        freeze_eigen=freeze_sl,
    )

    if epoch % 100 == 0:
        ckpt.save(log_path + f"/model-{epoch}", state)

        fig, error = evaluate1D(
            pinn,
            state.params,
            jnp.load(f"{cfg.DATA_PATH}/x.npy").reshape(-1, 1),
            cfg.DATA_PATH,
            Lc=cfg.Lc,
            xlim=cfg.DOMAIN[0],
        )
        metrics_tracker.register_figure(epoch, fig, "T-error")
        plt.close(fig)
        

        print(f"Epoch {epoch}, Loss: {weighted_loss:.4e}, error: {error:.4e},")

        metrics_tracker.register_scalars(
            epoch,
            names=["loss/weighted"]
            + [f"loss/{term}" for term in loss_terms]
            + [f"weight/{term}" for term in loss_terms]
            + ["error"] + ["sl"],
            values=[
                weighted_loss,
                *loss_components,
                *weight_components,
                error,
                state.params["params"]["sl"][0],
            ]
        )

        metrics_tracker.flush()
        


end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")