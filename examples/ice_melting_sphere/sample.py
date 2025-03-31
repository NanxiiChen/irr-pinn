import jax
import jax.numpy as jnp
from jax import random, vmap

from pinn import lhs_sampling, shifted_grid


class Sampler:

    def __init__(
        self,
        n_samples,
        domain=((-0.4, 0.4), (-0.4, 0.4), (0, 0.4), (0, 1)),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
            "model": None,
            "state": None,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def adaptive_sampling(self, residual_fn):
        key, self.key = random.split(self.key)
        adaptive_base = lhs_sampling(
            self.mins,
            self.maxs,
            self.adaptive_kw["num"] * self.adaptive_kw["ratio"],
            key=key,
        )
        residuals = residual_fn(adaptive_base)
        max_residuals, indices = jax.lax.top_k(
            jnp.abs(residuals), self.adaptive_kw["num"]
        )
        return adaptive_base[indices]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples, self.n_samples*3],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_pde_rar(self):
        key, self.key = random.split(self.key)
        # batch = shifted_grid(
        #     self.mins,
        #     self.maxs,
        #     [self.n_samples, self.n_samples, self.n_samples, self.n_samples * 3],
        #     key,
        # )
        batch = jnp.concatenate(self.sample_pde(), axis=-1)

        def residual_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["params"]
            x, t = batch[:, :-1], batch[:, -1:]
            return vmap(model.net_pde, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(residual_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[1][1], self.domain[2][1]],
            num=10000,
            key=key,
        )
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample(self, num_batch=None):
        data = (
            self.sample_pde_rar(),
            self.sample_ic(),
            self.sample_pde(),
        )
        if num_batch is None:
            return data


        batched_data = []
        for batch in data:
            x, t = batch
            # 使用array_split可以处理不能整除的情况
            x_batches = jnp.array_split(x, num_batch)[:num_batch]
            t_batches = jnp.array_split(t, num_batch)[:num_batch]
            batched_data.append(list(zip(x_batches, t_batches)))
        
        # 转置操作：将按数据类型组织的批次变成按批次编号组织的数据类型
        transposed_batched_data = list(zip(*batched_data))
        return transposed_batched_data
                    
            
        
            

