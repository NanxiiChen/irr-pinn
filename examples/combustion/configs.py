import jax.numpy as jnp

class Config:
    EPOCHS = 100000
    N_SAMPLES = 500
    ADAPTIVE_SAMPLES = 8000
    ADAPTIVE_BASE_RATE = 10
    NUM_BATCH = 10
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 200
    STAGGER_PERIOD = 25
    EMB_SCALE = 2.0
    EMB_DIM = 64

    DOMAIN = ((0, 1.0),)
    DATA_PATH = "./data/combustion/"
    LOG_DIR = "/root/tf-logs"
    PREFIX = "combustion/irr"
    TS = [0.000, 2.0000, 4.000, 5.000]

    NUM_LAYERS = 4
    HIDDEN_DIM = 64
    OUT_DIM = 1

    ACT_NAME = "tanh"
    ARCH_NAME = "mlp"
    FOURIER_EMB = False
    IRR = True

    W = 28.97e-3  # gas molecular weight, kg/mol
    LAMBDA = 2.6e-2  # thermal conductivity, W/(m-K)
    CP = 1000.0  # heat capacity, J/(kg-K)
    QF = 5.0e7  # fuel calorific value, J/kg
    R = 8.3145  # universal gas constant, J/(mol-K)
    A = 1.4e8  # pre-exponential factor
    EA = 1.214172e5  # activation energy, J/mol
    NU = 1.6  # reaction order
    RG = R / W  # gas constant, J/(kg-K)
    PHI = 0.4
    YF_IN = PHI / (4.0 + PHI)
    P_IN = 101325 * 1.0
    DTDX_IN = 1.0e5
    T_IN = 298
    RHO_IN = P_IN / (RG * T_IN)

    Lc = 1.5e-3  # xc = x / Lc
    PRE_SCALE = 1e6

    CAUSAL_CONFIGS = {
        "eps": 1e-2,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 10,
        "chunks": 24,
    }

