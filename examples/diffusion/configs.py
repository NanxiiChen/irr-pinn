import math
class Config:
    EPOCHS = 50000
    N_SAMPLES = 15
    ADAPTIVE_SAMPLES = 500
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 500
    SAVE_EVERY = 50
    EMB_SCALE = (2.0, 2.0)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[-1.0, 1.0], [-1.0, 1.0], [0, 1.0]]
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    DATA_PATH = "./data/diffusion/"
    PREFIX = "diffusion/irr"
    RESUME = None
    TS = [0.0, 0.25, 0.50, 0.75, 1.0]

    NUM_LAYERS = 8
    HIDDEN_DIM = 64
    OUT_DIM = 1

    ACT_NAME = "swish"
    ARCH_NAME = "modified_mlp"
    OPTIMIZER = "adam"
    FOURIER_EMB = False
    CAUSAL_WEIGHT = False
    IRR = True

    D = 0.1
    ALPHA = 0.5
    SIGMA = 0.15

    Lc = 1.0
    Tc = 1.0
    PDE_PRE_SCALE = 1.0

    CAUSAL_CONFIGS = {
        "eps": 1e-2,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 1e-3,
        "chunks": 24,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():
#         # 将所以的item作为全局变量，key = value 的形式
#         if not key.startswith("__"):
#             globals()[key] = value
