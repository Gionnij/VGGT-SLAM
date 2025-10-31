from dataclasses import dataclass, field
from typing import List, Dict, Union

from Config.config_def import NestedConfig


@dataclass
class AnchorGeneratorConfig(NestedConfig):
    ANGLES: List[List[int]] = field(default_factory=lambda: [[-90, 0, 90]])
    ASPECT_RATIOS: List[List[float]] = field(default_factory=lambda: [[0.5, 1.0, 2.0]])
    NAME: str = "DefaultAnchorGenerator"
    OFFSET: float = 0.0
    SIZES: List[List[int]] = field(default_factory=lambda: [[32, 64, 128, 256, 512]])


@dataclass
class BackboneConfig(NestedConfig):
    name: str = "resnet50"
    output_stride: int = 52
    pretrained_backbone: bool = False
    freeze: int = 0
    backbone_file: str = "~/Experiments/checkpoints/resnet50_hollistic.pth"


@dataclass
class FPNConfig(NestedConfig):
    FUSE_TYPE: str = "sum"
    IN_FEATURES: List[str] = field(default_factory=list)
    NORM: str = ""
    OUT_CHANNELS: int = 256


@dataclass
class MaskLossConfig(NestedConfig):
    name: str = "mask2former"
    args: Dict[str, Union[bool, float, int]] = field(
        default_factory=lambda: {
            "deep_supervision": True,
            "eos_coef": 0.1,
            "class_weight": 2.0,
            "dice_weight": 5.0,
            "mask_weight": 5.0,
            "dec_layers": 10,
            "num_points": 12544,
            "oversample_ratio": 3.0,
            "importance_sample_ratio": 0.75,
            "weight_dict_method": "power",
        }
    )
    weight: int = 1


@dataclass
class TestConfig(NestedConfig):
    OBJECT_MASK_THRESHOLD: float = 0.8
    OVERLAP_THRESHOLD: float = 0.8
    PANOPTIC_ON: bool = True
    INSTANCE_ON: bool = False
    SEMANTIC_ON: bool = True
    SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE: bool = False


@dataclass
class ClassEmbedConfig(NestedConfig):
    name: str = "linear"
    args: Dict = field(default_factory=lambda: {})


@dataclass
class ModelArgsConfig(NestedConfig):
    DETECTIONS_PER_IMAGE: int = 100
    DEC_LAYERS: int = 10
    DIM_FEEDFORWARD: int = 2048
    DROPOUT: float = 0.0
    ENC_LAYERS: int = 0
    ENFORCE_INPUT_PROJ: bool = False
    HIDDEN_DIM: int = 256
    NHEADS: int = 8
    NUM_OBJECT_QUERIES: int = 200
    PRE_NORM: bool = False
    SIZE_DIVISIBILITY: int = 32
    TRANSFORMER_DECODER_NAME: str = "MultiScaleMaskedTransformerDecoder"
    TRANSFORMER_IN_FEATURE: str = "multi_scale_pixel_decoder"
    TEST: TestConfig = field(default_factory=lambda: TestConfig())
    losses: Dict[str, MaskLossConfig] = field(
        default_factory=lambda: {"mask": MaskLossConfig()}
    )
    split_rows: int = 2
    class_embed: ClassEmbedConfig = field(default_factory=lambda: ClassEmbedConfig())
    mask_embed_type: str = "binary"
    vis_embeding: bool = True
    dropout_mlp: float = 0.0


@dataclass
class SEM_SEG_HEADConfig(NestedConfig):
    ASPP_CHANNELS: int = 256
    ASPP_DILATIONS: List[int] = field(default_factory=lambda: [6, 12, 18])
    ASPP_DROPOUT: float = 0.1
    COMMON_STRIDE: int = 4
    CONVS_DIM: int = 256
    DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: List[str] = field(
        default_factory=lambda: ["res3", "res4", "res5"]
    )
    DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS: int = 8
    DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS: int = 4
    IGNORE_VALUE: int = 255
    IN_FEATURES: List[str] = field(
        default_factory=lambda: ["res2", "res3", "res4", "res5"]
    )
    LOSS_TYPE: str = "hard_pixel_mining"
    LOSS_WEIGHT: float = 1.0
    MASK_DIM: int = 256
    NAME: str = "MaskFormerHead"
    NORM: str = "GN"
    NUM_CLASSES: int = 133
    PIXEL_DECODER_NAME: str = "MSDeformAttnPixelDecoder"
    PROJECT_CHANNELS: List[int] = field(default_factory=lambda: [48])
    PROJECT_FEATURES: List[str] = field(default_factory=lambda: ["res2"])
    TRANSFORMER_ENC_LAYERS: int = 6
    USE_DEPTHWISE_SEPARABLE_CONV: bool = False


@dataclass
class ModelConfig(NestedConfig):
    name: str = "mask2former"
    DETECTIONS_PER_IMAGE: int = 100
    ANCHOR_GENERATOR: AnchorGeneratorConfig = field(
        default_factory=lambda: AnchorGeneratorConfig()
    )
    backbone: BackboneConfig = field(default_factory=lambda: BackboneConfig())
    DEVICE: str = "cuda"
    FPN: FPNConfig = field(default_factory=lambda: FPNConfig())
    KEYPOINT_ON: bool = False
    LOAD_PROPOSALS: bool = False
    args: ModelArgsConfig = field(default_factory=lambda: ModelArgsConfig())
    MASK_ON: bool = False
    META_ARCHITECTURE: str = "MaskFormer"
    SEM_SEG_HEAD: SEM_SEG_HEADConfig = field(
        default_factory=lambda: SEM_SEG_HEADConfig()
    )


@dataclass
class DatamoduleArgsConfigCoco(NestedConfig):
    batch_size: int = 32
    crop_size: List[int] = field(default_factory=lambda: [512, 512])
    base_size: List[int] = field(default_factory=lambda: [512, 512])
    base_size_val: List[int] = field(default_factory=lambda: [512, 512])
    scale_range: List[int] = field(default_factory=lambda: [0.1, 2.0])
    target_type: List[str] = field(default_factory=lambda: ["semantic"])
    data_dir: str = "~/Datasets/Coco"
    val_batch_size: int = 1
    collate_name: str = "mask2former"
    panoptic_preprocessing: str = "mask2former"
    flip: bool = True
    persistent_workers: bool = False
    anomaly: bool = False
    keep_ar: bool = False


@dataclass
class DatamoduleConfig(NestedConfig):
    name: str = "coco"
    args: DatamoduleArgsConfigCoco = field(
        default_factory=lambda: DatamoduleArgsConfigCoco()
    )


@dataclass
class OptimizerArgsConfig(NestedConfig):
    lr: float = 0.0001
    weight_decay: float = 0.05


@dataclass
class LRSchedulerConfig(NestedConfig):
    type: str = "multistep"
    args: Dict[str, Union[float, int]] = field(default_factory=lambda: {})


@dataclass
class OptimizerConfig(NestedConfig):
    type: str = "adamw"
    args: OptimizerArgsConfig = field(default_factory=lambda: OptimizerArgsConfig())
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda:  LRSchedulerConfig())
    WEIGHT_DECAY_NORM: float = 0.0
    WEIGHT_DECAY_EMBED: float = 0
    backbone_lr_factor: float = 0.1


@dataclass
class TrainerArgsConfig(NestedConfig):
    gradient_clip_val: float = 0.01
    gradient_clip_algorithm: str = "norm"
    accelerator: str = "auto"
    strategy: str = "ddp_find_unused_parameters_true"


@dataclass
class CheckpointConfig(NestedConfig):
    monitor: str = "val_iou"
    mode: str = "max"
    save_last: bool = True


@dataclass
class TrainerConfig(NestedConfig):
    trainer_args: TrainerArgsConfig = field(default_factory=lambda: TrainerArgsConfig())
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    max_epochs: int = 50
    val_interval: int = 1
    val_check_interval: float = 1.0
    num_sanity_val_steps: int = 0
    gpus: int = 1
    tracking_uri: str = (
        ""
    )
    ckpt_path: str = None
    profiler: str = "simple"


@dataclass
class ModuleConfig(NestedConfig):
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())
    debug: bool = False
    full_frequency: int = 1
    small_validation_size: int = 100


@dataclass
class DefaultConfig(NestedConfig):
    CUDNN_BENCHMARK: bool = False
    datamodule: DatamoduleConfig = field(default_factory=lambda: DatamoduleConfig())
    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())
    experiment_name: str = "exp"
    type: str = "panoptic"
