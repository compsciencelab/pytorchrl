__version__ = "2.2.10"

from collections import namedtuple

__all__ = [
    "ALGORITHM", "EPISODES", "SCHEME", "TIME", "INFO_KEYS",
    "COLLECTION", "GRADIENT", "UPDATE", "OP_KEYS",
    "VERSION", "WEIGHTS", "NUMSAMPLES",
    "SYNC", "ASYNC", "CENTRAL", "PARALLEL", "FPS", "PL", "GA",
    "ACTOR", "STORAGE", "ENV_TRAIN", "ENV_TEST", "COMPONENT_KEYS",
    "OBS", "RHS", "DONE", "ACT", "REW", "OBS2", "RHS2", "DONE2",
    "DataTransitionKeys", "DataTransition", "OffPolicyDataKeys", "OnPolicyDataKeys",
]


# Information
ALGORITHM = "Algorithm"
EPISODES = "Episodes"
SCHEME = "Scheme"
TIME = "Time"
INFO_KEYS = (ALGORITHM, EPISODES, SCHEME, TIME)

# Operations
COLLECTION = "DataCollection"
GRADIENT = "GradientCompute"
UPDATE = "ActorUpdate"
PROCESSING = "DataProcessing"  # Includes Gradient computations + model Updates
CPRATIO = "DataCollection_DataProcessing_Ratio"
OP_KEYS = (COLLECTION, GRADIENT, UPDATE)

# Training
VERSION = "ActorVersion"
WEIGHTS = "ActorWeights"
NUMSAMPLES = "NumberSamples"

# Training Architecture
CENTRAL = "Central"
PARALLEL = "Parallel"
FPS = "FramesPerSecond"
PL = "PolicyLag"
GA = "GradientAsynchrony"
SYNC = "synchronous"
ASYNC = "asynchronous"

# Agent
ACTOR = "Actor"
STORAGE = "Storage"
ENV_TRAIN = "TrainEnvironment"
ENV_TEST = "TestEnvironment"
COMPONENT_KEYS = (ALGORITHM, ACTOR, STORAGE, ENV_TRAIN, ENV_TEST)

# DATA
OBS = "Observation"
RHS = "RecurrentHiddenStates"
DONE = "Done"
ACT = "Action"
ACTPROBS = "ActionProbs"
REW = "Reward"
IREW = "IntrinsicReward"
OBS2 = "NextObservation"
RHS2 = "NextRecurrentHiddenStates"
DONE2 = "NextDone"
VAL = "Value"
IVAL = "IntrinsicValue"
LOGP = "LogProbability"
ADV = "Advantage"
IADV = "IntrinsicAdvantage"
RET = "ExternalReturn"
IRET = "IntrinsicReturn"
MASK = "MaskedSamples"
INFO = "EnvironmentInformation"
DataTransitionKeys = (OBS, RHS, DONE, ACT, REW, OBS2, RHS2, DONE2, INFO)
DataTransition = namedtuple('DataTransition', DataTransitionKeys)
OffPolicyDataKeys = (OBS, RHS, DONE, ACT, REW, IREW, OBS2, RHS2, DONE2, ACTPROBS)
OnPolicyDataKeys = (OBS, RHS, DONE, ACT, REW, IREW, RET, IRET, VAL, IVAL, LOGP, ADV, IADV)
DemosDataKeys = (OBS, ACT, REW)

# DATA TYPES
float32 = "float32"
float16 = "float16"
int16 = "int16"
int8 = "int8"
uint8 = "uint8"

DEFAULT_DTYPES = {
    OBS: float32,
    RHS: float32,
    DONE: float32,
    ACT: float32,
    ACTPROBS: float32,
    REW: float32,
    OBS2: float32,
    RHS2: float32,
    DONE2: float32,
    VAL: float32,
    LOGP: float32,
    ADV: float32,
    RET: float32,
}

# ALGORITHMS
A2C = "A2C"
PPO = "PPO"
RND_PPO = "RND_PPO"
SAC = "SAC"
MPO = "MPO"
TD3 = "TD3"
DDPG = "DDPG"
DDQN = "DDQN"
