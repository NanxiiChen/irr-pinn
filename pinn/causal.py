import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt


class CausalWeightor:
    def __init__(
        self,
        num_chunks: int,
        t_range: tuple,
        **kwargs,
    ):

        self.num_chunks = num_chunks
        self.t_range = t_range
        # self.bins = jnp.linspace(t_range[0], t_range[1], num_chunks + 1)
        self.bins = kwargs.get("bins", jnp.linspace(t_range[0], t_range[1], num_chunks + 1))

    @partial(jax.jit, static_argnums=(0,))
    def compute_causal_weight(self, loss_chunks: jnp.array, eps: jnp.array):
        weights = []
        for chunk_id in range(self.num_chunks):
            if chunk_id == 0:
                weights.append(1.0)
            else:
                weights.append(jnp.exp(-eps * jnp.sum(loss_chunks[:chunk_id])))

        return jax.lax.stop_gradient(jnp.array(weights))

    @partial(jax.jit, static_argnums=(0,))
    def compute_causal_loss(self, residuals: jnp.array, t: jnp.array, eps: jnp.array):
        t_idx = jnp.digitize(t.flatten(), self.bins) - 1

        sum_residuals_sq = jax.ops.segment_sum(
            residuals**2, t_idx, num_segments=self.num_chunks
        )
        count_residuals = jax.ops.segment_sum(
            jnp.ones_like(residuals), t_idx, num_segments=self.num_chunks
        )
        loss_chunks = sum_residuals_sq / (count_residuals + 1e-12)

        causal_weights = self.compute_causal_weight(loss_chunks, eps)
        causal_loss = jnp.dot(loss_chunks, causal_weights)
        return causal_loss, {
            "causal_weights": causal_weights,
            "loss_chunks": loss_chunks,
        }

    def plot_causal_info(self, causal_weights, loss_chunks, eps):

        bins = (self.bins[1:] + self.bins[:-1]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        ax = axes[0]
        ax.plot(bins, causal_weights, marker="o")
        ax.set(
            xlabel="Time chunks",
            ylabel="Causal weights",
        )

        ax = axes[1]
        ax.plot(bins, loss_chunks, marker="o")
        ax.set(
            xlabel="Time chunks",
            ylabel="Causal loss",
        )

        fig.suptitle(f"EPS: {eps:.2e}")

        return fig

    def update_causal_eps(
        self,
        eps, 
        causal_weight,
        causal_configs,
    ):
        new_eps = eps
        if (
            causal_weight[-1] > causal_configs["max_last_weight"]
            and eps < causal_configs["max_eps"]
        ):
            # new_causal_configs["eps"] = (
            #     causal_configs["eps"] * causal_configs["step_size"]
            # )
            # print(f"Inc. eps to {causal_configs['eps']}")
            
            new_eps = eps * causal_configs["step_size"]
            print(f"Inc. eps to {eps}")


        if jnp.mean(causal_weight) < causal_configs["min_mean_weight"]:
            # new_causal_configs["eps"] = (
            #     causal_configs["eps"] / causal_configs["step_size"]
            # )
            # print(f"Dec. eps to {causal_configs['eps']}")
            new_eps = eps / causal_configs["step_size"]
            print(f"Dec. eps to {eps}")

        return new_eps


if __name__ == "__main__":
    t = jnp.array([0.0, 0.1, 0.7, 0.9, 0.3, 0.5, 1.0]).reshape(-1, 1)
    t_range = (0.0, 1.0)
    num = 4
    weightor = CausalWeightor(num, t_range)
    print(weightor._split_t(t))
