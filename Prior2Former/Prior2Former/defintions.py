from enum import Enum
from typing import TypeVar

from torch.utils.data import Dataset, Subset


class TaskDef(str, Enum):
    classification = "classification"
    semanticSegmentation = "semSeg"
    objectDetection = "object2Ddet"
    zeroclassification = "Zclassification"


class PreProcessingDef(str, Enum):
    norm = "norm"
    pretrained_norm = "preNorm"
    local_maxed = "local"
    equalized = "equalized"
    none = "none"


class ActiveLearningTrainingStrategies(str, Enum):
    retrain = "retrain"
    reuse = "reuse"
    continuous = "continuous"


class ActiveLearningScenario(str, Enum):
    stream = "Stream"
    pool = "Pool"
    streamBatch = "StreamBatch"
    poolStream = "PoolStream"
    multi_stream_pool = "MultiStreamPool"
    full = "Full"
    zeroShot = "ZeroShot"
    select = "Select"
    dataPruning = "DataPruning"


class Scenario(str, Enum):
    unc = "Uncertainty"
    ood = "Out-of-Distribution"
    al = "Active-Learning"
    train = "train"
    eval = "eval"

class DataUpdateScenario(str, Enum):
    standard = "standard"
    osal = "osal"
    osal_extending = "osal-extending"
    osal_near_far = "osal-near-far"
    osal_near_discovery = "osal-near-discovery"


class Engine(str, Enum):
    pytorch = "pytorch"
    pl = "pytorchLightning"


SubDataset = TypeVar("SubDataset", bound=Dataset)
SubSubset = TypeVar("SubSubset", bound=Subset)
