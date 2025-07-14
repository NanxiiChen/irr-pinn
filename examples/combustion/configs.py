import jax.numpy as jnp

class Config:
    EPOCHS = 100000
    N_SAMPLES = 500
    ADAPTIVE_SAMPLES = 250
    ADAPTIVE_BASE_RATE = 10
    NUM_BATCH = 10
    LR = 1e-3
    DECAY = 0.95
    DECAY_EVERY = 1000
    EMB_SCALE = 4
    EMB_DIM = 64
    DIM = 1

    DOMAIN = ((0.0, 1.0),)
    DATA_PATH = "./data/combustion/"
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    PREFIX = "combustion/noirr"
    RESUME = None
    # RESUME = "/root/autodl-tmp/tf-logs/combustion/irr/2025-07-13-22-19-34/model-5000/"

    NUM_LAYERS = 8
    HIDDEN_DIM = 32
    OUT_DIM = 1

    ACT_NAME = "tanh"
    ARCH_NAME = "resnet"
    OPTIMIZER = "rprop"
    FOURIER_EMB = False
    IRR = False
    RAR = False

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
    T_ADIA = T_IN + QF * YF_IN / CP

    Lc = 1.5e-3  # xc = x / Lc
    PRE_SCALE = 1e6
    T_PRE_SCALE = T_ADIA - T_IN