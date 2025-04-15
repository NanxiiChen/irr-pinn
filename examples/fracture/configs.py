class Config:
    EPOCHS = 100000
    N_SAMPLES = 15
    ADAPTIVE_SAMPLES = 5000
    ADAPTIVE_BASE_RATE = 5
    LR = 1e-4
    DECAY = 0.9
    DECAY_EVERY = 2000
    STAGGER_PERIOD = 100
    EMB_SCALE = (0.5, 0.5)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[-0.5, 0.5], [-0.5, 0.5], [0, 0.8]]
    DIM = 2
    DATA_PATH = "./data/fracture/"
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    PREFIX = "fracture/irr"
    RESUME = None
    # RESUME = "/root/tf-logs/fracture/irr/baseline-5/model-8000/"
    # TS = [0.0000, 0.3000, 0.7000, 0.7400, 0.7800]
    TS = [0.0000, 0.2000, 0.4000, 0.6000, 0.8000]

    NUM_LAYERS = 6
    HIDDEN_DIM = 200
    OUT_DIM = 3

    ACT_NAME = "gelu"
    ARCH_NAME = "modified_mlp"
    ASYMMETRIC = True
    FOURIER_EMB = True
    CAUSAL_WEIGHT = True
    IRR = True

    GC = 2.7
    L = 0.015
    UR = 0.007
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
        "max_eps": 1,
        "chunks": 24,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():self.adaptive_kw["num"]
#         if not key.startswith("__"):
#             globals()[key] = value
