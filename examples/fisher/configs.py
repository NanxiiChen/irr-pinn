import math

class Config:
    EPOCHS = 50000
    N_SAMPLES = 20
    ADAPTIVE_SAMPLES = 500
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 500
    SAVE_EVERY = 50
    EMB_SCALE = (5.0,)  # emb sacle for (x, t)
    EMB_DIM = 32

    DOMAIN = [[-2.0, 2.0], [0, 3.0]]
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    DATA_PATH = "./data/fisher/"
    PREFIX = "fisher/irr"
    RESUME = None
    TS = [0.0, 0.1, 0.3, 0.7, 1.0]

    NUM_LAYERS = 4
    HIDDEN_DIM = 40
    OUT_DIM = 1

    ACT_NAME = "swish"
    ARCH_NAME = "modified_mlp"
    OPTIMIZER = "adam"
    FOURIER_EMB = True
    CAUSAL_WEIGHT = False
    IRR = True
    RAR = True

    SIGMA = 1.0
    PHI = -1.0
    PSI = 0.0
    NU = 0.1
    g = lambda x, t: 0.0 * x

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
