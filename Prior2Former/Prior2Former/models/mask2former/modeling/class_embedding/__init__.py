from .linear_activation import DPNHead, Linear, DPNHead_BinaryNNO


def get_class_embedding(name: str, args: dict):
    if name == "linear":
        return Linear(**args)
    elif name == "dpn":
        return DPNHead(**args)
    elif name == "dpn_binary":
        return DPNHead_BinaryNNO(**args)
    else:
        raise NotImplementedError(f"Class Embedding of type {name} is not known")
