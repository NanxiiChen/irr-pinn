import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec

from examples.fracture.configs import Config as cfg


def evaluate2D(pinn, params, mesh, ref_path, ts, **kwargs):
    # fig, axes = plt.subplots(len(ts), 2, figsize=(10, 3*len(ts)))
    fig = plt.figure(figsize=(4 * len(ts), 14))
    gs = GridSpec(6, len(ts), height_ratios=[1, 0.05] * 3)
    vmin, vmax = kwargs.get("val_range", (0, 1))
    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim", (-0.5, 0.5))
    Lc = kwargs.get("Lc", 1.0)
    Tc = kwargs.get("Tc", 1.0)
    error_phi = 0
    error_ux = 0
    error_uy = 0
    mesh /= Lc
    for idx, tic in enumerate(ts):
        disp_uy = cfg.loading(tic)
        t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
        pred_phi, pred_disp = vmap(
            lambda x, t: pinn.net_u(params, x, t), in_axes=(0, 0)
        )(mesh, t)
        pred_phi = pred_phi[:, 0]
        pred_ux = pred_disp[:, 0]
        pred_uy = pred_disp[:, 1]

        ###################### PHI ######################
        ax = plt.subplot(gs[0, idx])
        cont_phi = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=pred_phi,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=r"$\phi$" + ", " + f"disp.: {disp_uy:.4f}",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )
        ax = plt.subplot(gs[1, idx])
        plt.colorbar(cont_phi, cax=ax, orientation="horizontal")

        ###################### UX ######################
        ax = plt.subplot(gs[2, idx])
        cont_ux = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=pred_ux,
            cmap="jet",
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=r"$u_x$",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )
        ax = plt.subplot(gs[3, idx])
        plt.colorbar(cont_ux, cax=ax, orientation="horizontal")

        ###################### UY ######################
        ax = plt.subplot(gs[4, idx])
        cont_uy = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=pred_uy,
            cmap="jet",
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=r"$u_y$",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )
        ax = plt.subplot(gs[5, idx])
        plt.colorbar(cont_uy, cax=ax, orientation="horizontal")

    # fig.tight_layout()
    return fig, (error_phi, error_ux, error_uy)


# def evaluate2D(pinn, params, mesh, ref_path, ts, **kwargs):
#     # fig, axes = plt.subplots(len(ts), 2, figsize=(10, 3*len(ts)))
#     fig = plt.figure(figsize=(4 * len(ts), 24))
#     gs = GridSpec(12, len(ts), height_ratios=[1, 0.05, 1, 0.05] * 3)
#     vmin, vmax = kwargs.get("val_range", (0, 1))
#     xlim = kwargs.get("xlim", (-0.5, 0.5))
#     ylim = kwargs.get("ylim", (-0.5, 0.5))
#     Lc = kwargs.get("Lc", 1.0)
#     Tc = kwargs.get("Tc", 1.0)
#     error_phi = 0
#     error_ux = 0
#     error_uy = 0
#     compute_fem_time = lambda t: 0.78 / jnp.tanh(5) * jnp.tanh(5 * t)

#     mesh /= Lc
#     for idx, tic in enumerate(ts):
#         fem_time = compute_fem_time(tic)
#         t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
#         pred_phi, pred_disp = vmap(
#             lambda x, t: pinn.net_u(params, x, t), in_axes=(0, 0)
#         )(mesh, t)
#         pred_phi = pred_phi[:, 0]
#         pred_ux = pred_disp[:, 0]
#         pred_uy = pred_disp[:, 1]

#         ###################### PHI ######################
#         ax = plt.subplot(gs[0, idx])
#         cont_phi = ax.scatter(
#             mesh[:, 0],
#             mesh[:, 1],
#             c=pred_phi,
#             cmap="jet",
#             vmin=vmin,
#             vmax=vmax,
#         )
#         ax.set(
#             xlabel="x",
#             ylabel="y",
#             title=f"t={tic}: " + r"$\phi$",
#             xlim=xlim,
#             ylim=ylim,
#             aspect="equal",
#         )
#         ax = plt.subplot(gs[1, idx])
#         plt.colorbar(cont_phi, cax=ax, orientation="horizontal")

#         ref_phi = jnp.load(f"{ref_path}/phi-{fem_time:.4f}.npy")
#         ax = plt.subplot(gs[2, idx])
#         error_bar_phi = ax.scatter(
#             mesh[:, 0],
#             mesh[:, 1],
#             c=jnp.abs(pred_phi - ref_phi),
#             cmap="jet",
#         )
#         ax.set(
#             xlabel="x",
#             ylabel="y",
#             title=r"$\phi$, $L^2=$" + f"{jnp.mean((pred_phi - ref_phi) ** 2):.2e}",
#             xlim=xlim,
#             ylim=ylim,
#             aspect="equal",
#         )
#         ax = plt.subplot(gs[3, idx])
#         # the ticks of the colorbar are in .2f format
#         plt.colorbar(error_bar_phi, cax=ax, orientation="horizontal")
#         ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

#         ###################### UX ######################
#         ax = plt.subplot(gs[4, idx])
#         cont_ux = ax.scatter(
#             mesh[:, 0],
#             mesh[:, 1],
#             c=pred_ux,
#             cmap="jet",
#         )
#         ax.set(
#             xlabel="x",
#             ylabel="y",
#             title=f"t={tic} : " + r"$u_x$",
#             xlim=xlim,
#             ylim=ylim,
#             aspect="equal",
#         )
#         ax = plt.subplot(gs[5, idx])
#         plt.colorbar(cont_ux, cax=ax, orientation="horizontal")

#         ref_ux = jnp.load(f"{ref_path}/u-{fem_time:.4f}.npy")[::2]
#         ax = plt.subplot(gs[6, idx])
#         error_bar_ux = ax.scatter(
#             mesh[:, 0],
#             mesh[:, 1],
#             c=jnp.abs(pred_ux - ref_ux),
#             cmap="jet",
#         )
#         ax.set(
#             xlabel="x",
#             ylabel="y",
#             title=r"$u_x$, $L^2=$" + f"{jnp.mean((pred_ux - ref_ux) ** 2):.2e}",
#             xlim=xlim,
#             ylim=ylim,
#             aspect="equal",
#         )

#         ax = plt.subplot(gs[7, idx])
#         plt.colorbar(error_bar_ux, cax=ax, orientation="horizontal")
#         ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

#         ###################### UY ######################
#         ax = plt.subplot(gs[8, idx])
#         cont_uy = ax.scatter(
#             mesh[:, 0],
#             mesh[:, 1],
#             c=pred_uy,
#             cmap="jet",
#         )
#         ax.set(
#             xlabel="x",
#             ylabel="y",
#             title=f"t={tic} : " + r"$u_y$",
#             xlim=xlim,
#             ylim=ylim,
#             aspect="equal",
#         )
#         ax = plt.subplot(gs[9, idx])
#         plt.colorbar(cont_uy, cax=ax, orientation="horizontal")

#         ref_uy = jnp.load(f"{ref_path}/u-{fem_time:.4f}.npy")[1::2]
#         ax = plt.subplot(gs[10, idx])
#         error_bar_uy = ax.scatter(
#             mesh[:, 0],
#             mesh[:, 1],
#             c=jnp.abs(pred_uy - ref_uy),
#             cmap="jet",
#         )
#         ax.set(
#             xlabel="x",
#             ylabel="y",
#             title=r"$u_y$, $L^2=$" + f"{jnp.mean((pred_uy - ref_uy) ** 2):.2e}",
#             xlim=xlim,
#             ylim=ylim,
#             aspect="equal",
#         )

#         ax = plt.subplot(gs[11, idx])
#         plt.colorbar(error_bar_uy, cax=ax, orientation="horizontal")
#         ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

#         error_phi += jnp.mean((pred_phi - ref_phi) ** 2)
#         error_ux += jnp.mean((pred_ux - ref_ux) ** 2)
#         error_uy = jnp.mean((pred_uy - ref_uy) ** 2)

#     error_phi /= len(ts)
#     error_ux /= len(ts)
#     error_uy /= len(ts)
#     fig.tight_layout()
#     return fig, (error_phi, error_ux, error_uy)
