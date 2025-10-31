from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional

from Config.config_def import NestedConfig


@dataclass
class PanopticArgsConfig:
    small_instance_area: int = 4048
    small_instance_weight: int = 3
    sigma: int = 4
    per_class_centers: bool = False


@dataclass
class DatamoduleArgsConfig(NestedConfig):
    batch_size: int = 30
    val_batch_size: int = 1
    crop_size: List[int] = field(default_factory=lambda: [512, 256])
    base_size: List[int] = field(default_factory=lambda: [512, 256])
    scale_range: List[int] = field(default_factory=lambda: [1.0, 1.0])
    persistent_workers: bool = False
    target_type: List[str] = field(default_factory=lambda: ["semantic", "instance"])
    panoptic_preprocessing: str = "u3hs"
    data_dir: str = "~/Datasets/Cityscapes"
    panoptic_args: PanopticArgsConfig = field(
        default_factory=lambda: PanopticArgsConfig()
    )
    flip: bool = True


@dataclass
class DatamoduleConfig(NestedConfig):
    name: str = "cityscapes"
    args: DatamoduleArgsConfig = field(default_factory=lambda: DatamoduleArgsConfig())


@dataclass
class BackboneConfig:
    name: str = "resnet50"
    output_stride: int = 16
    pretrained_backbone: bool = False
    backbone_file: str = "~/Experiments/model_zoo/resnet50-19c8e357.pth"
    freeze: int = -1


@dataclass
class AutoencoderConfig:
    name: str = "ae"
    nc: int = 2048
    height: int = 16
    width: int = 32
    z_dim: int = 32
    freeze: int = -1


@dataclass
class SemanticBranchConfig:
    feed_to_embedder: bool = True
    feed_to_detector: bool = True
    detach: bool = False
    type: str = "dpn"
    args: Dict[str, Union[str, bool, float]] = field(
        default_factory=lambda: {"nonlinearity": "softplus"}
    )


@dataclass
class DecoderArgsConfig(NestedConfig):
    aspp_channels: int = 256
    feature_dim: int = 8
    extra_detection_decoder: bool = True
    spatial_clustering: bool = False
    add_position: bool = False
    predict_sigmas: bool = True
    sigma_nonlinearity: str = "softplus"
    extra_thing_prototypes: bool = True
    rgb_branch: Optional[Dict[str, Union[str, bool, float]]] = None
    semantic_branch: SemanticBranchConfig = field(
        default_factory=lambda: SemanticBranchConfig()
    )
    per_class_centers: bool = False


@dataclass
class LossConfig:
    name: str = "mse"
    args: Dict[str, Union[str, bool, float]] = field(
        default_factory=lambda: {"reduction": "mean"}
    )
    weight: float = 1.0


@dataclass
class ModelArgsConfig(NestedConfig):
    in_channels: int = 2048
    feature_key: str = "res5"
    decoder_channels: int = 256
    atrous_rates: List[int] = field(default_factory=lambda: [6, 12, 18])
    low_level_channels: List[int] = field(default_factory=lambda: [512, 256])
    low_level_key: List[str] = field(default_factory=lambda: ["res3", "res2"])
    low_level_channels_project: List[int] = field(default_factory=lambda: [64, 32])
    split_rows: int = 3
    certainty_stats: bool = True
    decoder_args: DecoderArgsConfig = field(default_factory=lambda: DecoderArgsConfig())
    losses: Dict[str, LossConfig] = field(default_factory=lambda: {})


@dataclass
class ModelConfig(NestedConfig):
    name: str = "u3hs"
    backbone: BackboneConfig = field(default_factory=lambda: BackboneConfig())
    autoencoder: AutoencoderConfig = field(default_factory=lambda: AutoencoderConfig())
    args: ModelArgsConfig = field(default_factory=lambda: ModelArgsConfig())
    bn_momentum_backbone: float = 0.02
    bn_momentum_decoder: float = 0.02


@dataclass
class OptimizerArgsConfig(NestedConfig):
    lr: float = 0.005


@dataclass
class LrSchedularConfig(NestedConfig):
    type: str = "polynomial"
    args: Dict[str, Union[float, int]] = field(default_factory=lambda: {})


@dataclass
class OptimizerConfig:
    args: OptimizerArgsConfig = field(default_factory=lambda: OptimizerArgsConfig())
    type: str = "adam"
    # (
    # Dict)[str, Union[str, bool, float]] = field(
    # default_factory=lambda: {"lr": 0.005}
    # )
    backbone_lr_factor: float = 0.05
    lr_scale: float = 1.0
    lr_scheduler: LrSchedularConfig = field(default_factory=lambda: LrSchedularConfig())


@dataclass
class PostProcessConfig:
    threshold: float = 0.1
    nms_kernel: int = 7
    top_k_instance: int = 250
    spatial_clustering: bool = False
    bias_weight: float = 0.0
    mixture_dist: str = "normal"
    predict_sigmas: bool = True
    instance_class: str = "majority_vote"
    semantic_uncertainty: str = "dirichlet_strength"
    unnormalized_dists: bool = True
    certainty_threshold: Optional[float] = 0.7


@dataclass
class ModuleConfig(NestedConfig):
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())
    post_process: PostProcessConfig = field(default_factory=lambda: PostProcessConfig())
    max_validation_imgs: int = 16


@dataclass
class CheckpointConfig:
    monitor: str = "validation_pq"
    mode: str = "max"
    every_n_train_steps: Optional[int] = None
    save_top_k: int = 1
    auto_insert_metric_name: bool = True


@dataclass
class TrainerConfig(NestedConfig):
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    callbacks: List[Dict] = field(default_factory=lambda: [])
    trainer_args: Dict[str, Union[str, bool, float]] = field(default_factory=lambda: {})
    max_epochs: int = 120
    val_check_interval: Union[float, int] = 1.0
    val_interval: int = 10
    num_sanity_val_steps: int = 0
    gpus: int = 1
    tracking_uri: str = (
        "file:/Experiments/mlflow"
    )


@dataclass
class DefaultConfig(NestedConfig):
    type: str = "panoptic"
    datamodule: DatamoduleConfig = field(default_factory=lambda: DatamoduleConfig())
    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())
    experiment_name: str = "exp"
    run_name: str = ""



