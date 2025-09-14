from jax import numpy as jnp


class Config:
    EPOCHS = 100000
    N_SAMPLES = 15
    ADAPTIVE_SAMPLES = 500
    ADAPTIVE_BASE_RATE = 6
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 1000
    STAGGER_PERIOD = 25
    EMB_SCALE = (2.0, 2.0)  # emb sacle for (x, t)
    EMB_DIM = 128

    DOMAIN = [[-0.5, 0.5], [-0.5, 0.5], [0, 1.0]]
    DIM = 2
    DATA_PATH = "./data/fracture/"
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    PREFIX = "fracture/irr"
    RESUME = None
    # RESUME = "/root/autodl-tmp/tf-logs/fracture/irr/0913-saved-stage/model-89000/"
    # TS = [0.0000, 0.3000, 0.7000, 0.7400, 0.7800]
    TS = [0.0000, 0.2500, 0.5000, 0.7500, 1.0000]

    NUM_LAYERS = 6
    HIDDEN_DIM = 100
    OUT_DIM = 3

    ACT_NAME = "swish" # tanh, swish, snake...
    ARCH_NAME = "modified_mlp" # mlp, modified_mlp, moe
    OPTIMIZER = "adam"
    CHANGE_OPT_AT = 1000000
    FOURIER_EMB = False
    CAUSAL_WEIGHT = True
    IRR = True
    POINT_WISE_WEIGHT = False   # 有两种形式，1/(alpha + grad(phi)) 或者 exp(-grad(phi)*alpha)
    RAR = True   # RAR 和PWW实际上是相反作用，RAR强调界面，PWW弱化界面
    DEAD_POINTS_WEIGHT = False
    FREEZE = False

    GC = 2.7
    L = 0.024
    UR = 0.00525
    LOAD_ON_DIR = "y"
    LOAD_ON = 1 if LOAD_ON_DIR == "y" else 0
    LAMBDA = 121.1538e3
    MU = 80.7692e3
    NU = 0.3

    Lc = 1.0
    Tc = 1.0
    DISP_PRE_SCALE = 1e3
    STRESS_PRE_SCALE = 1e4
    PF_PRE_SCALE = 1e2
    PF_EPS = 0.0

    CAUSAL_CONFIGS = {
        "stress_x_eps": 1e-2,
        "stress_y_eps": 1e-2,
        "stress_eps": 1e-2,
        "pf_eps": 50,
        "energy_eps": 1e-2,
        "step_size": 5,
        "max_last_weight": 0.90,
        "min_mean_weight": 0.3,
        "max_eps": 50,
        "chunks": 10,
    }

    @classmethod
    def loading(cls, t, alpha=4.0):
        # return cls.UR * t
        return cls.UR / jnp.tanh(alpha) * jnp.tanh(alpha * t)
    

    @classmethod
    def loading_reverse(cls, loading, alpha=4.0):
        # give loading, return t
        return jnp.arctanh(
            loading * jnp.tanh(alpha) / cls.UR
        ) / alpha


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():self.adaptive_kw["num"]
#         if not key.startswith("__"):
#             globals()[key] = value

