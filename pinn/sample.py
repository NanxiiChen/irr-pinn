
import jax
from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt


def mesh_flat(*args):
    return [coord.reshape(-1, 1) for coord in jnp.meshgrid(*args)]


def lhs_sampling(mins, maxs, num, key):
    dim = len(mins)
    u = (jnp.arange(0, num) + 0.5) / num  # 每个区间的中点

    keys = random.split(key, dim)
    result = jnp.zeros((num, dim))

    for i in range(dim):
        perm = random.permutation(keys[i], u)
        result = result.at[:, i].set(mins[i] + perm * (maxs[i] - mins[i]))

    return result


def shifted_grid(mins, maxs, nums, key, eps=1e-3):
    dim = len(mins)
    mins = jnp.array(mins)
    maxs = jnp.array(maxs)
    nums = jnp.array(nums)

    grids = []
    distances = (maxs - mins) / (nums - 1)

    keys = random.split(key, dim)
    shifts = jnp.array(
        [
            random.uniform(keys[i], shape=(), minval=-distances[i], maxval=distances[i])
            for i in range(dim)
        ]
    )

    # 创建带偏移的网格
    for i in range(dim):
        grid_i = jnp.linspace(mins[i], maxs[i], nums[i]) + shifts[i]
        grid_i = jnp.clip(grid_i, mins[i] + eps, maxs[i] - eps)
        grids.append(grid_i)

    # 使用mesh_flat创建网格点
    data = jnp.stack(mesh_flat(*grids), axis=-1).reshape(-1, dim)
    # shuffle
    data = data[random.permutation(key, data.shape[0])]
    return data


class Sampler:

    def __init__(
        self,
        n_samples,
        domain=((-0.5, 0.5), (0, 0.5), (0, 1)),
        key=None,
        adaptive_kw=None,
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.key = key if key is not None else random.PRNGKey(0)
        self.adaptive_kw = (
            adaptive_kw
            if adaptive_kw is not None
            else {
                "ratio": 10,
                "num": 5000,
            }
        )
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples * 3],
            key,
        )
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
            _, indices = jax.lax.top_k(jnp.abs(res), self.adaptive_kw["num"])
            selected_points = adaptive_base[indices]
            rar_points = rar_points.at[
                idx * self.adaptive_kw["num"] : (idx + 1) * self.adaptive_kw["num"], :
            ].set(selected_points)


        data = jnp.concatenate(
            [
                common_points,
                rar_points,
            ],
            axis=0,
        )
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        raise NotImplementedError("Initial condition sampling is not implemented.")

    def sample_bc(self):
        raise NotImplementedError("Boundary condition sampling is not implemented.")

    def sample_flux(self):
        raise NotImplementedError("Flux sampling is not implemented.")

    def sample(self, *args, **kwargs):
        return (
            self.sample_pde_rar(*args, **kwargs),
            self.sample_ic(),
            self.sample_bc(),
            self.sample_flux(),
            self.sample_pde(),
        )



if __name__ == "__main__":
    mins = jnp.array([0, 0])
    maxs = jnp.array([1, 1])
    num = 100

    # 使用JAX的随机数生成器
    key = random.PRNGKey(42)
    key1, key2 = random.split(key)

    # 生成样本
    data = shifted_grid(mins, maxs, [20, 20], key1)
    data2 = shifted_grid(mins, maxs, [20, 20], key2)

    # 可视化
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label="Sample 1")
    plt.scatter(data2[:, 0], data2[:, 1], alpha=0.6, label="Sample 2")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Latin Hypercube Sampling (n={num})")
    plt.show()
