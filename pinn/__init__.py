from matplotlib import RcParams
import matplotlib
import jax

from .arch import MLP, ModifiedMLP
from .evaluator import evaluate1D, evaluate2D, evaluate3D
from .metrics import MetricsTracker
from .sample import mesh_flat, lhs_sampling, shifted_grid
from .causal import CausalWeightor, update_causal_eps
from .utils import StaggerSwitch
from .model import PINN

jax.config.update("jax_default_matmul_precision", "high")


matplotlib.use("Agg")
RcParams.update(
    {
        "font.size": 16,
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.family": "sans-serif",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
