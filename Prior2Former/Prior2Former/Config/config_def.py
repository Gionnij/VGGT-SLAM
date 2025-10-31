from dataclasses import dataclass, asdict
from utils.configmap import ConfigMap


@dataclass
class Config(ConfigMap):
    experiment_config: ConfigMap
    training_config: ConfigMap

@dataclass
class NestedConfig():
    pass

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

@dataclass
class Unionable:
    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))