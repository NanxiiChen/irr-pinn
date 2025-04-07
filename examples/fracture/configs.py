"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""


class Config:
    EPOCHS = 2000
    N_SAMPLES = 15
    ADAPTIVE_SAMPLES = 20000
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 200
    STAGGER_PERIOD = 25
    EMB_SCALE = (2.0, 2.0)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[-0.5, 0.5], [-0.5, 0.5], [0, 0.78]]
    DATA_PATH = "./data/fracture/"
    LOG_DIR = "/root/tf-logs"
    PREFIX = "fracture/irr"
    TS = [0.0000, 0.5000, 0.7200, 0.7800]

    NUM_LAYERS = 6
    HIDDEN_DIM = 64
    OUT_DIM = 3

    ACT_NAME = "tanh"
    ARCH_NAME = "mlp"
    ASYMMETRIC = True
    FOURIER_EMB = True
    CAUSAL_WEIGHT = True
    IRR = True

    GC = 2.7
    L = 0.015
    UR = 0.007
    LAMBDA = 121.1538e3
    MU = 80.7692e3

    Lc = 1.0
    Tc = 1.0
    DISP_PRE_SCALE = 1e2
    STRESS_PRE_SCALE = 1e6
    PF_PRE_SCALE = 1e3

    CAUSAL_CONFIGS = {
        "stress_eps": 1e-2,
        "pf_eps": 1e-5,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 1e2,
        "chunks": 24,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():
#         # 将所以的item作为全局变量，key = value 的形式
#         if not key.startswith("__"):
#             globals()[key] = value
