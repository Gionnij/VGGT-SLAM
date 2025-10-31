from collections import OrderedDict

import torch


def apply_nonlinearity(nonlinearity: str, tensor: torch.Tensor):
    if nonlinearity == "exp":
        return tensor.exp()
    elif nonlinearity == "relu":
        return tensor.clip(0)
    elif nonlinearity == "softplus":
        return torch.nn.Softplus()(tensor)
    else:
        raise Exception(f"Nonlinearity {nonlinearity} not known")


def copy_parameters(module1: torch.nn.Module, module2: torch.nn.Module):
    state_dict = {}
    for name, param in module1.named_parameters():
        state_dict[name] = param
    module2.load_state_dict(OrderedDict(state_dict), strict=False)
