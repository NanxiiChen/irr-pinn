from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import jit



def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    xdim = kwargs.get("xdim", 3)
    time_dependent = kwargs.get("time_dependent", True)
    if time_dependent:
        params = model.init(rng, jnp.ones(xdim), jnp.ones(1))
    else:
        params = model.init(rng, jnp.ones(xdim))
    opt_method = kwargs.get("optimizer", "adam")
    scheduler = optax.exponential_decay(lr, decay_every, decay, 
                                        staircase=False, 
                                        end_value=kwargs.get("end_value", 1e-5))
    if opt_method == "adam":
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(scheduler),
        )
    elif opt_method == "soap":
        from .optimizer import soap
        optimizer = soap(
            learning_rate=scheduler,
            b1=0.99,
            b2=0.999,
            precondition_frequency=2,
        )
    elif opt_method == "rprop":
        from .optimizer import rprop
        # RPROP不使用学习率调度器，而是自适应调整步长
        init_step_size = kwargs.get("init_step_size", lr)  # 可以使用传入的学习率作为初始步长
        eta_plus = kwargs.get("eta_plus", 1.2)
        eta_minus = kwargs.get("eta_minus", 0.5)
        step_size_min = kwargs.get("step_size_min", 1e-6)
        step_size_max = kwargs.get("step_size_max", 1.0)
        
        optimizer = rprop(
            init_step_size=init_step_size,
            eta_plus=eta_plus,
            eta_minus=eta_minus,
            step_size_min=step_size_min,
            step_size_max=step_size_max
        )
    elif opt_method == "lbfgs":
        from .optimizer import lbfgs
        
        maxiter = kwargs.get("lbfgs_maxiter", 20)
        history_size = kwargs.get("lbfgs_history_size", 10)
        tol = kwargs.get("lbfgs_tol", 1e-3)
        line_search = kwargs.get("lbfgs_line_search", "zoom")
        verbose = kwargs.get("lbfgs_verbose", False)
        
        optimizer = lbfgs(
            maxiter=maxiter,
            history_size=history_size,
            tol=tol,
            line_search=line_search,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_method}")

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jit, static_argnums=(0,))
def train_step(loss_fn, state, batch, eps):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(params, batch, eps)
    )
    # handle NaN or Inf values in gradients
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)



# @partial(jit, static_argnums=(0,))
def lbfgs_train_step(loss_fn, state, batch, eps):
    """LBFGS专用的训练步骤，支持完整的优化过程"""
    params = state.params
    
    # 为LBFGS创建值函数和梯度函数
    def value_fun(p):
        loss_val, _ = loss_fn(p, batch, eps)
        return loss_val
    
    def grad_fun(p):
        _, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, eps)[0])(p)
        return grads
    
    # 应用LBFGS优化器，传递值函数和梯度函数
    updates, new_opt_state = state.tx.update(
        None,  # LBFGS不使用传入的梯度
        state.opt_state,
        params,
        value_fun=value_fun,
        grad_fun=grad_fun
    )
    
    # 应用更新
    new_params = optax.apply_updates(params, updates)
    
    # 计算最终结果，用于返回损失组件
    weighted_loss, (loss_components, weight_components, aux_vars) = loss_fn(new_params, batch, eps)
    
    # 创建新状态
    new_state = train_state.TrainState(
        step=state.step + 1,
        apply_fn=state.apply_fn,
        params=new_params,
        tx=state.tx,
        opt_state=new_opt_state
    )
    
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)

class StaggerSwitch:
    def __init__(
        self,
        pde_names=[
            "ac",
            "ch",
        ],
        stagger_period=10,
    ):
        self.pde_names = pde_names
        self.stagger_period = stagger_period
        self.epoch = 0

    def step_epoch(self):
        self.epoch += 1

    def decide_pde(self):
        epoch_round = len(self.pde_names) * self.stagger_period
        idx = (self.epoch % epoch_round) // self.stagger_period
        return self.pde_names[idx]


if __name__ == "__main__":
    stagger_switch = StaggerSwitch()
    for i in range(50):
        print(i, stagger_switch.decide_pde())
        stagger_switch.step_epoch()
