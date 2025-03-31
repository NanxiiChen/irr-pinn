"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""

import math


class Config:
    EPOCHS = 100000
    N_SAMPLES = 20
    ADAPTIVE_SAMPLES = 8000
    ADAPTIVE_BASE_RATE = 10
    # MAX_BATCH_SIZE = 5000
    NUM_BATCH = 10
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 200
    STAGGER_PERIOD = 25
    EMB_SCALE = (2.0, 0.2)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0, 1.0))
    DATA_PATH = "./data/ice-melting/"
    LOG_DIR = "/root/tf-logs"
    PREFIX = "ice-melting/noirr"
    TS = [0.000, 2.0000, 4.000, 5.000]

    NUM_LAYERS = 4
    HIDDEN_DIM = 64
    OUT_DIM = 1

    ACT_NAME = "tanh"
    ARCH_NAME = "mlp"
    FOURIER_EMB = True
    CAUSAL_WEIGHT = True
    IRR = False

    LAMBDA = 5
    NN = 64
    HH = 100 / NN
    EPSILON = 6 * HH / (2 * math.sqrt(2) * math.atanh(0.9))
    MM = 0.1
    R0 = 35

    Lc = 100.0
    Tc = 5.0
    AC_PRE_SCALE = 1e6
    CH_PRE_SCALE = 1e0

    CAUSAL_CONFIGS = {
        "eps": 1e-2,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 10,
        "chunks": 24,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():
#         # 将所以的item作为全局变量，key = value 的形式
#         if not key.startswith("__"):
#             globals()[key] = value
