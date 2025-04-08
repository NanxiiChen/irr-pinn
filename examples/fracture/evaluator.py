import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec

from examples.ice_melting.configs import Config as cfg


def evaluate2D(pinn, params, mesh, ref_path, ts, **kwargs):
    # fig, axes = plt.subplots(len(ts), 2, figsize=(10, 3*len(ts)))
    fig = plt.figure(figsize=(4 * len(ts), 20))
    gs = GridSpec(8, len(ts), height_ratios=[1, 0.05, 1, 0.05] * 2)
    vmin, vmax = kwargs.get("val_range", (0, 1))
    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim", (-0.5, 0.5))
    Lc = kwargs.get("Lc", 1.0)
    Tc = kwargs.get("Tc", 1.0)
    error_phi = 0
    error_uy = 0
    mesh /= Lc
    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
        pred_phi, pred_disp = vmap(
            lambda x, t: pinn.net_u(params, x, t), in_axes=(0, 0)
        )(mesh, t)
        pred_phi = pred_phi[:, 0]
        pred_uy = pred_disp[:, 1]


        ax = plt.subplot(gs[0, idx])
        cont_phi = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=pred_phi,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=f"t={tic}: " + r"$\phi$",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )
        ax = plt.subplot(gs[1, idx])
        plt.colorbar(
            cont_phi,
            cax=ax,
            orientation="horizontal",
        )

        ax = plt.subplot(gs[4, idx])
        cont_uy = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=pred_uy,
            cmap="coolwarm",
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=f"t={tic} : " + r"$u_y$",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )
        ax = plt.subplot(gs[5, idx])
        plt.colorbar(
            cont_uy,
            cax=ax,
            orientation="horizontal",
        )


        ref_phi = jnp.load(f"{ref_path}/phi-{tic:.4f}.npy")
        ref_uy = jnp.load(f"{ref_path}/u-{tic:.4f}.npy")[1::2]

        ax = plt.subplot(gs[2, idx])
        error_bar_phi = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=jnp.abs(pred_phi - ref_phi),
            cmap="coolwarm",
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=f"t={tic}",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )

        ax = plt.subplot(gs[6, idx])
        error_bar_uy = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=jnp.abs(pred_uy - ref_uy),
            cmap="coolwarm",
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            title=f"t={tic}",
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
        )


        ax = plt.subplot(gs[3, idx])
        # the ticks of the colorbar are in .2f format
        plt.colorbar(error_bar_phi, cax=ax, orientation="horizontal")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        ax = plt.subplot(gs[7, idx])
        plt.colorbar(error_bar_uy, cax=ax, orientation="horizontal")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        error_phi += jnp.mean((pred_phi - ref_phi) ** 2)
        error_uy = jnp.mean((pred_uy - ref_uy) ** 2)



    plt.tight_layout()
    error_phi /= len(ts)
    error_uy /= len(ts)
    fig.tight_layout()
    return fig, (error_phi, error_uy)