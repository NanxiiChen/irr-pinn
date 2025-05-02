from jax import numpy as jnp


class Config:
    EPOCHS = 100000
    N_SAMPLES = 20
    ADAPTIVE_SAMPLES = 1000
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 500
    STAGGER_PERIOD = 25
    EMB_SCALE = (4, 4)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[-0.5, 0.5], [-0.5, 0.5], [0, 1.0]]
    DIM = 2
    DATA_PATH = "./data/fracture/"
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    PREFIX = "fracture/irr"
    RESUME = None
    # RESUME = "/root/autodl-tmp/tf-logs/fracture/irr/baseline-best0429-model400_6-epoch30k-actswish/model-30000/"
    # TS = [0.0000, 0.3000, 0.7000, 0.7400, 0.7800]
    TS = [0.0000, 0.2500, 0.5000, 0.8000, 1.0000]

    NUM_LAYERS = 6
    HIDDEN_DIM = 400
    OUT_DIM = 3

    ACT_NAME = "swish"
    ARCH_NAME = "modified_mlp"
    OPTIMIZER = "adam"
    CHANGE_OPT_AT = 5000000
    ASYMMETRIC = True
    FOURIER_EMB = False
    CAUSAL_WEIGHT = True
    IRR = True

    GC = 2.7
    L = 0.024
    UR = 0.0070
    LOAD_ON_DIR = "y"
    LOAD_ON = 1 if LOAD_ON_DIR == "y" else 0
    LAMBDA = 121.1538e3
    MU = 80.7692e3
    NU = 0.3

    Lc = 1.0
    Tc = 1.0
    DISP_PRE_SCALE = 1e2
    STRESS_PRE_SCALE = 1e4
    PF_PRE_SCALE = 1e2

    CAUSAL_CONFIGS = {
        "stress_x_eps": 1e-1,
        "stress_y_eps": 1e-1,
        "stress_eps": 1e-1,
        "pf_eps": 1e-1,
        "step_size": 5,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 50,
        "chunks": 12,
    }

    @classmethod
    def loading(cls, t, alpha=2.0):
        # return cls.UR * t
        return cls.UR / jnp.tanh(alpha) * jnp.tanh(alpha * t)


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():self.adaptive_kw["num"]
#         if not key.startswith("__"):
#             globals()[key] = value
