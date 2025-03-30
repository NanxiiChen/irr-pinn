from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, vmap

from pinn import CausalWeightor, MLP, ModifiedMLP


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
        causal_weightor: CausalWeightor = None,
    ):
        super().__init__()

        self.cfg = config

        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_irr,
        ]
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[self.cfg.ARCH_NAME](
            act_name=self.cfg.ACT_NAME,
            num_layers=self.cfg.NUM_LAYERS,
            hidden_dim=self.cfg.HIDDEN_DIM,
            out_dim=self.cfg.OUT_DIM,
            fourier_emb=self.cfg.FOURIER_EMB,
            emb_scale=self.cfg.EMB_SCALE,
            emb_dim=self.cfg.EMB_DIM,
        )
        self.causal_weightor = causal_weightor

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        # phi0 = self.ref_sol_ic(x, t)
        # return phi0 + self.model.apply(params, x, t) * t
        return nn.tanh(self.model.apply(params, x, t))

    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x, t):
        phi = self.net_u(params, x, t)
        dphi_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)[0] / self.cfg.Tc
        # note, if the `[0]` is not added, the shape of hess_x will be (1, 3, 3)
        # then the trace will be the sum of all the elements in the matrix
        # leading to totally wrong results, FUCK!!!
        hess_x = jax.hessian(self.net_u, argnums=1)(params, x, t)[0]
        lap_phi = jnp.linalg.trace(hess_x) / self.cfg.Lc**2

        Fphi = 0.25 * (phi**2 - 1) ** 2
        dFdphi = phi**3 - phi

        pde = (
            dphi_dt
            - self.cfg.MM * (lap_phi - dFdphi / self.cfg.EPSILON**2)
            + self.cfg.LAMBDA * jnp.sqrt(2 * Fphi) / self.cfg.EPSILON
        )
        return pde.squeeze()

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        raise NotImplementedError


    @partial(jit, static_argnums=(0,))
    def net_speed(self, params, x, t):
        dphi_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)[0]
        return dphi_dt

    @partial(jit, static_argnums=(0,))
    def loss_ic(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
        return jnp.mean((u - ref) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(jax.nn.relu(dphi_dt))

    @partial(jit, static_argnums=(0,))
    def loss_pde(self, params, batch, eps):
        x, t = batch
        res = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(res**2), {}
        else:
            return self.causal_weightor.compute_causal_loss(
                res,
                t,
                eps,
            )

    # @partial(jit, static_argnums=(0,))
    # def loss_pde(self, params, batch, eps):
    #     max_bs = self.cfg.MAX_BATCH_SIZE
    #     x, t = batch
    #     # since the row of the data might not be the integer times of max_bs
    #     # so the shape of the data might not be the same
    #     # we need to pad the data to make sure the shape is the same
    #     x_padded = jnp.pad(x, ((0, max_bs - x.shape[0] % max_bs), (0, 0)))
    #     t_padded = jnp.pad(t, ((0, max_bs - t.shape[0] % max_bs), (0, 0)))

    #     # shuffle the data
    #     idx = jax.random.permutation(random.PRNGKey(0), x_padded.shape[0])
    #     x_padded = x_padded[idx]
    #     t_padded = t_padded[idx]

    #     num_batches = x_padded.shape[0] // max_bs

    #     if not self.cfg.CAUSAL_WEIGHT:

    #         def process_batch(start):
    #             x_batch = jax.lax.dynamic_slice(x_padded, (start, 0), (max_bs, x_padded.shape[1]))
    #             t_batch = jax.lax.dynamic_slice(t_padded, (start, 0), (max_bs, t_padded.shape[1]))
    #             res = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x_batch, t_batch)
    #             loss = jnp.mean(res**2)
    #             return loss

    #         loss = 0
    #         for idx in range(0, num_batches):
    #             start = idx * max_bs
    #             loss += process_batch(start)
    #         loss /= num_batches
    #         return loss, {}
    #     else:
    #         raise NotImplementedError
    # return self.causal_weightor.compute_causal_loss(
    #     res,
    #     t,
    #     eps,
    # )

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch, eps):
        losses = []
        grads = []
        for idx, (loss_item_fn, batch_item) in enumerate(
            zip(self.loss_fn_panel, batch)
        ):
            if idx == 0:
                (loss_item, aux), grad_item = jax.value_and_grad(
                    loss_item_fn, has_aux=True, argnums=0
                )(params, batch_item, eps)
            else:
                loss_item, grad_item = jax.value_and_grad(loss_item_fn)(
                    params, batch_item
                )
            losses.append(loss_item)
            grads.append(grad_item)

        losses = jnp.array(losses)
        weights = self.grad_norm_weights(grads)
        # weights = jax.lax.stop_gradient(jnp.array([3.0, 1.0, 1.0]))
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)

        return jnp.sum(weights * losses), (losses, weights, aux)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-6):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)


                

