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

    def ref_sol_bc_top(self, x, t):
        # uy = 0.007 * 0.78 / np.tanh(3) * np.tanh(3 * t)
        return jax.lax.stop_gradient(self.cfg.loading(t[0]))

    def ref_sol_bc_bottom(self, x, t):
        return jax.lax.stop_gradient(jnp.array([0.0, 0.0, 0.0]))

    def ref_sol_bc_crack(self, x, t):
        # phi = exp(-|y| / l)
        phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L))
        return jax.lax.stop_gradient(phi)

    def ref_sol_ic_phi(self, x, t):
        phi = jnp.exp(-jnp.abs(x[1] / self.cfg.L)) * (1 - jax.nn.sigmoid(x[0] * 200))
        return jax.lax.stop_gradient(phi)

    # def loss_ic(self, params, batch, *args, **kwargs):
    #     x, t = batch
    #     phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
    #     phi = phi[:, 0]
    #     ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
    #     return jnp.mean((phi - ref) ** 2)
    def loss_ic_phi(self, params, batch, *args, **kwargs):
        x, t = batch
        phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        phi = phi[:, 0]
        ref = vmap(self.ref_sol_ic_phi, in_axes=(0, 0))(x, t)
        icphi = jnp.mean((phi - ref) ** 2)
        return icphi, {}

    def loss_ic_ux(self, params, batch, *args, **kwargs):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ux = disp[:, 0]
        icux = jnp.mean(ux**2) * self.cfg.DISP_PRE_SCALE**2
        return icux, {}

    def loss_ic_uy(self, params, batch, *args, **kwargs):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        uy = disp[:, 1]
        icuy = jnp.mean(uy**2) * self.cfg.DISP_PRE_SCALE**2
        return icuy, {}

    def loss_bc_bottom_phi(self, params, batch, *args, **kwargs):
        x, t = batch
        phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        phi = phi[:, 0]
        bottom = jnp.mean((phi) ** 2)
        return bottom, {}

    def loss_bc_bottom_ux(self, params, batch, *args, **kwargs):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ux = disp[:, 0]
        bottom = jnp.mean(ux**2) * self.cfg.DISP_PRE_SCALE**2
        return bottom, {}

    def loss_bc_bottom_uy(self, params, batch, *args, **kwargs):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        uy = disp[:, 1]
        bottom = jnp.mean(uy**2) * self.cfg.DISP_PRE_SCALE**2
        return bottom, {}

    def loss_bc_top_phi(self, params, batch, *args, **kwargs):
        x, t = batch
        phi, _ = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        phi = phi[:, 0]
        top = jnp.mean((phi) ** 2)
        return top, {}

    def loss_bc_top_uy(self, params, batch, *args, **kwargs):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_bc_top, in_axes=(0, 0))(x, t)
        top = jnp.mean((disp[:, self.cfg.LOAD_ON] - ref) ** 2)  * self.cfg.DISP_PRE_SCALE**2
        return top, {}

    def loss_bc_top_ux(self, params, batch, *args, **kwargs):
        x, t = batch
        _, disp = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        top = jnp.mean(disp[:, 1 - self.cfg.LOAD_ON] ** 2)  * self.cfg.DISP_PRE_SCALE**2
        return top, {}

    def loss_bc_crack(self, params, batch, *args, **kwargs):
        x, t = batch
        phi = vmap(
            lambda x, t: self.net_u(params, x, t)[0],
            in_axes=(0, 0),
        )(
            x, t
        )[:, 0]
        ref = vmap(self.ref_sol_bc_crack, in_axes=(0, 0))(x, t)
        crack = jnp.mean((phi - ref) ** 2)

        return crack, {}

    def loss_bc_right(self, params, batch, *args, **kwargs):
        x, t = batch
        nabla_phi_fn = jax.jacrev(
            lambda params, x, t: self.net_u(params, x, t)[0], argnums=1
        )
        dphi_dx = vmap(
            lambda params, x, t: nabla_phi_fn(params, x, t)[0, 0], in_axes=(None, 0, 0)
        )(params, x, t)
        return jnp.mean(dphi_dx**2), {}

    def loss_bc_sigmax(self, params, batch, *args, **kwargs):
        x, t = batch
        norm_vector = jnp.array([1.0, 0.0])
        phi = vmap(
            lambda params, x, t: self.net_u(params, x, t)[0], in_axes=(None, 0, 0)
        )(params, x, t).squeeze()
        sigma = vmap(
            self.sigma, in_axes=(None, 0, 0)
        )(params, x, t)
        traction_vectors = jnp.einsum('ijk,k->ij', sigma, norm_vector) # shape [batch, 2]
        # res = traction_vectors * (1 - phi)**2  # traction in x direction
        phi = jnp.expand_dims(phi, axis=-1)  # shape [batch, 1]
        res = traction_vectors * (1 - phi)**2
        return jnp.mean(res**2), {}
        # res = jnp.sum(jnp.abs(res), axis=-1)
        # return jnp.mean(res**2), {}

    def loss_pf_energy(self, params, batch, *args, **kwargs):
        x, t = batch
        energy_fn = self.net_pf_energy
        pf_energy = vmap(
            lambda params, x, t: energy_fn(params, x, t), in_axes=(None, 0, 0)
        )(params, x, t)
        return jnp.mean(pf_energy**2), {}
    


# causal_first_point = 0.3
# causal_bins_tail = jnp.linspace(
#     causal_first_point, cfg.DOMAIN[-1][-1], cfg.CAUSAL_CONFIGS["chunks"]
# )
# # insert 0.0 at the beginning of the bins
# causal_bins = jnp.insert(causal_bins_tail, 0, 0.0)



# bins = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.94, 0.98, 1.0])
causal_weightor = CausalWeightor(
    cfg.CAUSAL_CONFIGS["chunks"],
    t_range=cfg.DOMAIN[-1],
    # bins=bins,
)

loss_terms = [
    "pde",
    "ic_phi",
    "ic_ux",
    "ic_uy",
    "bc_bottom_phi",
    "bc_bottom_ux",
    "bc_bottom_uy",
    "bc_top_phi",
    "bc_top_ux",
    "bc_top_uy",
    "bc_crack",
    "bc_sigmax",
    "irr",
    # "complementarity"
    # "irr_pf",
]

pinn = FracturePINN(config=cfg, causal_weightor=causal_weightor, loss_terms=loss_terms)

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
    end_value=1e-6
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

stagger = StaggerSwitch(
    pde_names=["stress_y", "stress_x", "pf"], 
    stagger_period=cfg.STAGGER_PERIOD
)


start_time = time.time()
for epoch in range(cfg.EPOCHS):

    if epoch == cfg.CHANGE_OPT_AT:
        print("Change optimizer to soap")
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

    # if epoch % cfg.STAGGER_PERIOD == 0:
    if epoch % 10 == 0:
        batch = sampler.sample(
            fns=[pinn.psi],
            # fns=[getattr(pinn, f"net_{pde_name}"),],
            params=state.params,
            rar=pinn.cfg.RAR,
            model=pinn
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
        if epoch % 500 == 0:
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

        # replace `pde` with `pde_name` in `loss_terms`
        loss_terms_epoch = loss_terms.copy()
        loss_terms_epoch[0] = pde_name
        metrics_tracker.register_scalars(
            epoch,
            names=["loss/weighted"]
            + [f"loss/{term}" for term in loss_terms_epoch]
            + [f"weight/{term}" for term in loss_terms_epoch],
            values=[
                weighted_loss,
                *loss_components,
                *weight_components,
            ],
        )

        if cfg.CAUSAL_WEIGHT and epoch % 500 == 0:
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
