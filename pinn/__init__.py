import os
import matplotlib
from matplotlib import font_manager, rcParams

from .arch import MLP, ModifiedMLP
from .causal import CausalWeightor
from .evaluator import evaluate1D, evaluate2D, evaluate3D
from .metrics import MetricsTracker
# from .model import PINN
from .sample import lhs_sampling, mesh_flat, shifted_grid, Sampler
from .utils import StaggerSwitch
from .train import train_step, create_train_state, StaggerSwitch



def configure_matplotlib():
    matplotlib.use("Agg")
    rcParams.update(
        {
            "font.size": 16,
            "font.family": "serif",
            "font.serif": ["Palatino Linotype"],
            "mathtext.fontset": "cm",
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def load_fonts(font_dir="./fonts/"):
    font_dir = os.path.abspath(font_dir)
    if not os.path.exists(font_dir):
        print(f"Warning: Font directory '{font_dir}' does not exist.")
        return
    font_names = os.listdir(font_dir)
    if not font_names:
        print(f"Warning: No fonts found in directory '{font_dir}'.")
        return
    for font_name in font_names:
        font_manager.fontManager.addfont(os.path.join(font_dir, font_name))


configure_matplotlib()
load_fonts()
