def get_detectron_backbone(name, cfg, input_shape = None):
    if input_shape is None:
        from ...layers import ShapeSpec
        input_shape = ShapeSpec(channels=3)
    if name == "build_resnet_backbone":
        from .resnet import get_resnet_backbone
        return get_resnet_backbone(cfg, input_shape)
    else:
        raise NotImplementedError(f"backbone {name} not implemented")