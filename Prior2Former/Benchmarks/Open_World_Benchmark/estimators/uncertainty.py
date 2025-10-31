import numpy as np
import torch

def resize(x, size=[512,1024]):
    return torch.nn.functional.interpolate(
        x, size=size, mode="bilinear", align_corners=False
    )

class Uncertainty:
    #for u3hs
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, model, x=None, out=None):
        resized = False
        if out is None:
            if list(x.shape[-2:]) == [1024,2048]: #e.g. for fishyscapes and L&F
                x = resize(x, size=[512,1024])
                resized = True
            out = model(x)
        
        uncertainties = []
        for i in range(out["semantic"].shape[0]):
            uncertainties.append(
                model.decoder.semantic_head.get_uncertainty(out, i).cpu()
            )
        uncertainties =  torch.stack(uncertainties)
        if resized:
            uncertainties = resize(uncertainties.unsqueeze(0), size=[1024,2048]).squeeze(0)
        uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
        return uncertainties
