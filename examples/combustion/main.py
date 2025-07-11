import datetime
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from jax import jit, random

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from examples.combustion import (
    PINN,
    CombustionSampler,
    evaluate1D,
    cfg,
)

from pinn import (
    CausalWeightor,
    MetricsTracker,
    train_step,
    create_train_state,
)

class CombustionPINN(PINN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_bc,
            self.loss_irr,
        ]

    