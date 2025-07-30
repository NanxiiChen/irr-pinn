import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec

def evaluate2D(pinn, params, mesh, ref_path, ts, **kwargs):
    fig = plt.figure(figsize=(4 * len(ts), 10))
    gs = GridSpec(3, len(ts), height_ratios=[1, 1, 0.05])

    xlim = kwargs.get('xlim', (-1, 1))
    ylim = kwargs.get('ylim', (-1, 1))
    error = 0
    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic
        pred = vmap(
            pinn.net_u, in_axes=(None, 0, 0)
        )(params, mesh, t).reshape(-1, 1)

        ax = fig.add_subplot(gs[0, idx])
        ax.scatter(
            mesh[:, 0], mesh[:, 1], 
            c=pred[:, 0], cmap='coolwarm'
        )
        ax.set(
            xlabel="x", ylabel="y",
            title=f"t = {tic:.2f}",
            xlim=xlim, ylim=ylim,
            aspect='equal'
        )

        ref_sol = jnp.load(f"{ref_path}/phi-{tic:.2f}.npy").reshape(-1, 1)

        ax = fig.add_subplot(gs[1, idx])
        error_bar = ax.scatter(
            mesh[:, 0], mesh[:, 1],
            c=jnp.abs(pred - ref_sol)[:, 0], cmap='coolwarm'
        )
        ax.set(
            xlabel="x", ylabel="y",
            title=f" t = {tic:.2f}",
            xlim=xlim, ylim=ylim,
            aspect='equal'
        )

        ax = fig.add_subplot(gs[2, idx])
        cbar = fig.colorbar(
            error_bar, cax=ax, orientation='horizontal',
        )
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        error += jnp.mean((pred - ref_sol) ** 2) / jnp.mean(ref_sol ** 2)

    plt.tight_layout()
    error /= len(ts)
    return fig, error
