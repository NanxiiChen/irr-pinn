import jax.numpy as jnp
from jax import random

from pinn import lhs_sampling, shifted_grid, Sampler


class DiffusionSampler(Sampler):
    def __init__(
        self,
        n_samples,
        domain=((-1.0, 1.0), (-1.0, 1.0), (0, 1)),
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
            [self.n_samples, self.n_samples, self.n_samples],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0]],
            maxs=[self.domain[0][1], self.domain[1][1]],
            num=self.n_samples**2*10,
            key=key,
        )
        x_local = lhs_sampling(
            mins=[-0.10, -0.10], maxs=[0.10, 0.10],
            num=self.n_samples**2 * 10, key=self.key
        )
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        key, self.key = random.split(self.key)
        xt = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]+0.05],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2,
            key=key,
        )
        top = jnp.concatenate([
            xt[:, 0:1],
            jnp.ones_like(xt[:, 0:1]) * self.domain[1][1],
            xt[:, 1:2],],
            axis=1
        )
        bottom = jnp.concatenate([
            xt[:, 0:1],
            jnp.ones_like(xt[:, 0:1]) * self.domain[1][0],
            xt[:, 1:2],],
            axis=1
        )

        yt = lhs_sampling(
            mins=[self.domain[1][0], self.domain[2][0]+0.05],
            maxs=[self.domain[1][1], self.domain[2][1]],
            num=self.n_samples**2,
            key=key,
        )
        left = jnp.concatenate([
            jnp.ones_like(yt[:, 0:1]) * self.domain[0][0],
            yt[:, 0:1],
            yt[:, 1:2],],
            axis=1
        )
        right = jnp.concatenate([
            jnp.ones_like(yt[:, 0:1]) * self.domain[0][1],
            yt[:, 0:1],
            yt[:, 1:2],],
            axis=1
        )
        data = jnp.concatenate([top, bottom, left, right], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample(self, *args, **kwargs):
        return (
            self.sample_pde(),
            self.sample_ic(),
            self.sample_bc(),
            self.sample_irr(),
        )

    def sample_irr(self):
        """直接在圆形区域内采样，避免浪费计算"""
        key, self.key = random.split(self.key)
        
        # 极坐标采样
        n_irr = self.n_samples**2  # 控制irr采样点数
        r_samples = jnp.sqrt(random.uniform(key, (n_irr,))) * 0.5  # [0, 0.5]
        theta_samples = random.uniform(key, (n_irr,)) * 2 * jnp.pi
        
        # 转换为笛卡尔坐标
        x = r_samples * jnp.cos(theta_samples)
        y = r_samples * jnp.sin(theta_samples)
        irr_x = jnp.stack([x, y], axis=1)
        
        # 时间采样
        t_samples = random.uniform(key, (n_irr, 1)) * (self.domain[2][1] - self.domain[2][0]) + self.domain[2][0]
        
        return irr_x, t_samples
