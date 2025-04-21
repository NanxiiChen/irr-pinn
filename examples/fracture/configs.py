from jax import numpy as jnp


class Config:
    EPOCHS = 100000
    N_SAMPLES = 15
    ADAPTIVE_SAMPLES = 2000
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 2000
    STAGGER_PERIOD = 50
    EMB_SCALE = (2.0, 2.0)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[-0.5, 0.5], [-0.5, 0.5], [0, 1.0]]
    DIM = 2
    DATA_PATH = "./data/fracture/"
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    PREFIX = "fracture/irr"
    RESUME = None
    # RESUME = "/root/autodl-tmp/tf-logs/fracture/irr/baseline3-model128_8-stage1-10k-linear_loading//model-10000/"
    # TS = [0.0000, 0.3000, 0.7000, 0.7400, 0.7800]
    TS = [0.0000, 0.2500, 0.5000, 0.8000, 1.0000]

    NUM_LAYERS = 6
    HIDDEN_DIM = 200
    OUT_DIM = 3

    ACT_NAME = "snake"
    ARCH_NAME = "modified_mlp"
    OPTIMIZER = "adam"
    CHANGE_OPT_AT = 2000
    ASYMMETRIC = True
    FOURIER_EMB = False
    CAUSAL_WEIGHT = True
    IRR = True

    GC = 2.7
    L = 0.024
    UR = 0.0065
    LAMBDA = 121.1538e3
    MU = 80.7692e3
    NU = 0.3

    Lc = 1.0
    Tc = 1.0
    DISP_PRE_SCALE = 1e2
    STRESS_PRE_SCALE = 1e5
    PF_PRE_SCALE = 1e2

    CAUSAL_CONFIGS = {
        "stress_eps": 1e-2,
        "pf_eps": 1e-2,
        "step_size": 5,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 50,
        "chunks": 12,
    }

    @classmethod
    def loading(cls, t):
        # return cls.UR * t
        return cls.UR / jnp.tanh(2.5) * jnp.tanh(2.5* t)


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():self.adaptive_kw["num"]
#         if not key.startswith("__"):
#             globals()[key] = value
