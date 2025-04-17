import jax
import jax.numpy as jnp
from jax import random, vmap

from pinn import lhs_sampling, shifted_grid, Sampler


class FractureSampler(Sampler):

    def __init__(
        self,
        n_samples,
        domain=((-0.5, 0.5), (0, 0.5), (0, 0.1)),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples * 2],
            key,
        )

        # data_crack = shifted_grid(
        #     [self.mins[0], self.mins[1], 0.70],
        #     self.maxs,
        #     [self.n_samples, self.n_samples, self.n_samples],
        #     key,
        # )

        # data = jnp.concatenate([data_global, data_crack], axis=0)

        return data[:, :-1], data[:, -1:]

    def sample_pde_rar(self, fns, params):
        key, self.key = random.split(self.key)
        grid_key, lhs_key = random.split(key)
        common_points = jnp.concatenate(self.sample_pde(), axis=-1)

        adaptive_base = lhs_sampling(
            self.mins,
            self.maxs,
            self.adaptive_kw["num"] * self.adaptive_kw["ratio"],
            key=lhs_key,
        )
        x, t = adaptive_base[:, :-1], adaptive_base[:, -1:]
        rar_points = jnp.zeros(
            (self.adaptive_kw["num"] * len(fns), adaptive_base.shape[1])
        )

        for idx, fn in enumerate(fns):
            res = jax.lax.stop_gradient(vmap(fn, in_axes=(None, 0, 0))(params, x, t))
            res = res.reshape(self.adaptive_kw["num"] * self.adaptive_kw["ratio"], -1)
            res = jnp.linalg.norm(res, ord=1, axis=-1)
            # if res.ndim > 1:
            #     res = jnp.linalg.norm(res, ord=1, axis=-1)
            _, indices = jax.lax.top_k(jnp.abs(res), self.adaptive_kw["num"])
            selected_points = adaptive_base[indices]
            rar_points = rar_points.at[
                idx * self.adaptive_kw["num"] : (idx + 1) * self.adaptive_kw["num"], :
            ].set(selected_points)

        data = jnp.concatenate([common_points, rar_points], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0]],
            maxs=[self.domain[0][1], self.domain[1][1]],
            num=self.n_samples**2,
            key=key,
        )

        x_local = lhs_sampling(
            mins=[self.domain[0][0], -0.1],
            maxs=[self.domain[0][1], 0.1],
            num=self.n_samples**2 * 5,
            key=self.key,
        )
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        key, self.key = random.split(self.key)
        xt = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2,
            key=key,
        )
        top = jnp.concatenate(
            [xt[:, 0:1], jnp.ones_like(xt[:, 0:1]) * self.domain[1][1], xt[:, 1:2]],
            axis=1,
        )
        bottom = jnp.concatenate(
            [xt[:, 0:1], jnp.ones_like(xt[:, 0:1]) * self.domain[1][0], xt[:, 1:2]],
            axis=1,
        )
        yt = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0]],
            maxs=[self.domain[0][1], self.domain[1][1]],
            num=self.n_samples**2,
            key=key,
        )
        right = jnp.concatenate(
            [jnp.ones_like(yt[:, 0:1]) * self.domain[0][1], yt[:, 0:1], yt[:, 1:2]],
            axis=1,
        )

        crack = lhs_sampling(
            mins=[self.domain[0][0], -0.05, self.domain[2][0]],
            maxs=[0.0, 0.05, self.domain[2][1]],
            num=self.n_samples**2,
            key=self.key,
        )

        return {
            "top": (top[:, :-1], top[:, -1:]),
            "bottom": (bottom[:, :-1], bottom[:, -1:]),
            "right": (right[:, :-1], right[:, -1:]),
            "crack": (crack[:, :-1], crack[:, -1:]),
        }

    def sample(self, *args, **kwargs):
        # rar = self.sample_pde_rar(*args, **kwargs)
        rar = self.sample_pde_rar(*args, **kwargs)
        return (
            rar,
            self.sample_ic(),
            self.sample_bc(),
            rar,
        )
