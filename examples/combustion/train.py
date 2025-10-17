from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, tree_util
import optax
from flax.training import train_state

def create_gradient_mask(params, freeze_model=False, freeze_eigen=False, eigen_tag="sl", model_tag="Dense"):
    def mask_fn(path, param):
        if freeze_model and model_tag in path:
            return jnp.zeros_like(param)
        if freeze_eigen and eigen_tag in path:
            return jnp.zeros_like(param)
        return jnp.ones_like(param)

    return tree_util.tree_map_with_path(mask_fn, params)



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
        optimizer = optax.adam(scheduler)

    elif opt_method == "soap":
        from pinn.optimizer import soap
        optimizer = soap(
            learning_rate=scheduler,
            b1=0.99,
            b2=0.999,
            precondition_frequency=2,
        )
    elif opt_method == "rprop":
        from pinn.optimizer import rprop
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
        from pinn.optimizer import lbfgs
        
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




@partial(jit, static_argnums=(0, 4, 5))
def train_step(loss_fn, state, batch, eps, freeze_model, freeze_eigen, **kwargs):
    params = state.params
    
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(params, batch, eps)
    )
    
    model_tag = kwargs.get("model_tag", "Dense")
    eigen_tag = kwargs.get("eigen_tag", "sl")
    grad_mask = create_gradient_mask(grads, freeze_model=freeze_model, freeze_eigen=freeze_eigen,
                                     model_tag=model_tag, eigen_tag=eigen_tag)
    masked_grads = tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)
    
    new_state = state.apply_gradients(grads=masked_grads)
    total_grad_norm = optax.global_norm(masked_grads)
    aux_vars.update({"total_grad_norm": total_grad_norm})
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)
