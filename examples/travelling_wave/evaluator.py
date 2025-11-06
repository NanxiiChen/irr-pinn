import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec


def evaluate1D(pinn, params, data_path, ts):
    mesh = jnp.load(f"{data_path}/mesh.npy")
    times = jnp.load(f"{data_path}/times.npy")
    sol = jnp.load(f"{data_path}/sol.npy")

    Lc = pinn.cfg.Lc
    Tc = pinn.cfg.Tc
    mesh = mesh / Lc
    times = times / Tc

    xx, tt = jnp.meshgrid(mesh, times)
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(2, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    xx_flat = xx.flatten()[:, None]
    tt_flat = tt.flatten()[:, None]
    pred_flat = vmap(
        pinn.net_u, in_axes=(None, 0, 0)
    )(params, xx_flat, tt_flat)
    pred = pred_flat.reshape(xx.shape)
    im0 = ax0.imshow(
        pred,
        extent=[mesh.min(), mesh.max(), times.min(), times.max()],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0
    )
    ax0.set_title("Prediction")

    ax1 = fig.add_subplot(gs[0, 1])
    diff = jnp.abs(sol - pred)
    error = jnp.mean(diff**2)
    im1 = ax1.imshow(
        diff,
        extent=[mesh.min(), mesh.max(), times.min(), times.max()],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0
    )
    ax1.set_title(f"Error (MSE={error:.2e})")

    ax2 = fig.add_subplot(gs[1, :])
    for idx, t in enumerate(ts):
        t_c = t / Tc
        t_idx = jnp.argmin(jnp.abs(times - t_c))
        ax2.plot(mesh, sol[t_idx, :], label=f"Ref t={t:.2f}")
        ax2.plot(mesh, pred[t_idx, :], "--", label=f"Pred t={t:.2f}")
    ax2.set_title("Time snapshots")
    ax2.legend()



    # # 创建混合3D和2D的子图
    # fig = plt.figure(figsize=(12, 15))
    # gs = GridSpec(3, 2, figure=fig)
    
    # # 前两个子图为3D
    # ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    # ax0.plot_surface(xx, tt, sol, cmap="viridis")
    # ax0.set_title("Reference (3D)")

    # # prediction
    # xx_flat = xx.flatten()[:, None]
    # tt_flat = tt.flatten()[:, None]
    # pred_flat = vmap(
    #     pinn.net_u, in_axes=(None, 0, 0)
    # )(params, xx_flat, tt_flat)

    # pred = pred_flat.reshape(xx.shape)
    # ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    # ax1.plot_surface(xx, tt, pred, cmap="viridis")
    # ax1.set_title("Prediction (3D)")
    
    # # 后两个子图为2D
    # ax2 = fig.add_subplot(gs[1, 0])
    # ax3 = fig.add_subplot(gs[1, 1])

    # error = jnp.mean((sol - pred) ** 2)
    # # sol_log = jnp.log10(sol + 1e-10)
    # # pred_log = jnp.log10(jnp.abs(pred) + 1e-10)
    # # error = jnp.mean(jnp.abs((sol_log - pred_log) / (sol_log + 1e-10)))

    # # plot the time-snapshots at 0, 0.25, 0.5, 0.75, 1.0 of the final time
    # times_indices = times.shape[0] * jnp.array(ts)
    # times_indices = times_indices.astype(jnp.int32).tolist()


    # # diffs = []
    # for idx, tic_idx in enumerate(times_indices):
    #     tic_show = times[tic_idx]
        
    #     # reference (2D)
    #     ax2.plot(mesh, sol[tic_idx, :], label=f"t={tic_show:.2f}")
    #     ax2.set_title("Reference (2D)")
    #     ax2.legend()

    #     # prediction (2D)
    #     tic_inp = jnp.ones_like(mesh) * tic_show
    #     pred_tic = vmap(
    #         pinn.net_u, in_axes=(None, 0, 0)
    #     )(params, mesh[:, None], tic_inp[:, None]).squeeze(-1)
    #     ax3.plot(mesh, pred_tic, label=f"t={tic_show:.2f}")
    #     ax3.set_title("Prediction (2D)")
    #     ax3.legend()

    #     # diff_tic = sol[tic_idx, :] - pred_tic
    #     # rel_error = jnp.linalg.norm(diff_tic) / jnp.linalg.norm(sol[tic_idx, :])
    #     # diffs.append(rel_error)

    # # error = jnp.mean(jnp.array(diffs))


    # ax4 = fig.add_subplot(gs[2, :])
    # mesh_inp = jnp.zeros_like(times)
    # pred = vmap(
    #     pinn.net_u, in_axes=(None, 0, 0)
    # )(params, mesh_inp[:, None], times[:, None]).squeeze(-1)
    # ax4.plot(times, pred, label="Prediction at x=0.0", ls="solid")
    # x0_idx = jnp.argmin(jnp.abs(mesh - 0.0))
    # ax4.plot(times, sol[:, x0_idx], label="Reference at x=0.0", ls="dashed")
    # ax4.set_title("Prediction vs Reference at x=0.0")
    # ax4.set_yscale("log")
    # ax4.legend()



    return fig, error
