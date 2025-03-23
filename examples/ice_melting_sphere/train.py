from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import jit


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones(3), jnp.ones(1))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(scheduler),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jit, static_argnums=(0,))
def train_step(loss_fn, state, batch, eps):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(params, batch, eps)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux)


@partial(jit, static_argnums=(0, 4))  # 将 num_batch 作为静态参数
def train_step_minibatch(
    loss_fn,
    state,
    batch,
    eps,
    num_batch,  # num_batch 是静态参数
):
    # 初始化累积梯度为零
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_carry = (init_grads, None)

    def body_fn(batch_idx, carry):
        # carry: (accumulated_grads, None)
        accumulated_grads, _ = carry

        # 使用 jax.lax.dynamic_index_in_dim 进行动态索引
        cur_batch = jax.lax.dynamic_index_in_dim(batch, batch_idx, keepdims=False)

        # 计算当前 batch 的梯度和损失
        (weighted_loss, (loss_components, weight_components, aux)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(
                state.params, cur_batch, eps
            )
        )

        # 累积梯度
        accumulated_grads = jax.tree_util.tree_map(
            lambda acc, g: acc + g, accumulated_grads, grads
        )

        # 返回累积的梯度和最后一个 batch 的其他值
        return (accumulated_grads, (weighted_loss, loss_components, weight_components, aux))

    # 使用 jax.lax.fori_loop 遍历所有 mini-batches
    final_carry = jax.lax.fori_loop(
        0, num_batch, body_fn, init_carry
    )

    # 平均化累积的梯度
    accumulated_grads = jax.tree_util.tree_map(
        lambda g: g / num_batch, final_carry[0]
    )

    # 更新模型参数
    new_state = state.apply_gradients(grads=accumulated_grads)

    # 返回新的状态和最后一个 batch 的损失和其他值
    return new_state, final_carry[1]

