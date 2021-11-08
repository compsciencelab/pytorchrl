__version__ = "1.4.14"

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
OBS2 = "NextObservation"
RHS2 = "NextRecurrentHiddenStates"
DONE2 = "NextDone"
VAL = "Value"
LOGP = "LogProbability"
ADV = "Advantage"
RET = "Return"
DataTransitionKeys = (OBS, RHS, DONE, ACT, REW, OBS2, RHS2, DONE2)
DataTransition = namedtuple('DataTransition', DataTransitionKeys)
OffPolicyDataKeys = (OBS, RHS, DONE, ACT, REW, OBS2, RHS2, DONE2, ACTPROBS)
OnPolicyDataKeys = (OBS, RHS, DONE, ACT, REW, RET, VAL, LOGP, ADV)
DemosDataKeys = (OBS, ACT, REW)

# ALGORITHMS
A2C = "A2C"
PPO = "PPO"
SAC = "SAC"
MPO = "MPO"
TD3 = "TD3"
DDPG = "DDPG"
DDQN = "DDQN"

# -----------------------------------------------------------------------------

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
