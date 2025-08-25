import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec


def evaluate2D(pinn, params, mesh, ref_path, ts, **kwargs):
    # fig, axes = plt.subplots(len(ts), 2, figsize=(10, 3*len(ts)))
    fig = plt.figure(figsize=(10, 3 * len(ts)))
    gs = GridSpec(len(ts), 3, width_ratios=[1, 1, 0.05])
    vmin, vmax = kwargs.get("val_range", (0, 1))
    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim", (0, 0.5))
    Lc = kwargs.get("Lc", 1e-4)
    Tc = kwargs.get("Tc", 10.0)
    error = 0
    mesh /= Lc
    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
        pred = vmap(lambda x, t: pinn.net_u(params, x, t)[0], in_axes=(0, 0))(
            mesh, t
        ).reshape(mesh.shape[0], 1)

        ax = plt.subplot(gs[idx, 0])
        ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=pred[:, 0],
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

        ref_sol = jnp.load(f"{ref_path}/sol-{tic:.3f}.npy")[:, 0:1]

        ax = plt.subplot(gs[idx, 1])
        error_bar = ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            c=jnp.abs(pred - ref_sol),
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
        # colorbar for error

        ax = plt.subplot(gs[idx, 2])
        # the ticks of the colorbar are in .2f format
        plt.colorbar(error_bar, cax=ax)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        error += jnp.mean((pred - ref_sol) ** 2)

    plt.tight_layout()
    error /= len(ts)
    return fig, error
