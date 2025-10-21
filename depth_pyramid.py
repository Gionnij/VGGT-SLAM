# depth_pyramid.py
import torch
import torch.nn as nn

class DepthDPTBackbone(nn.Module):
    """
    Wraps VGGT depth_head to expose intermediate DPT feature maps.
    """
    def __init__(self, vggt):
        super().__init__()
        self.vggt = vggt
        self.cache = {}
        self.handles = []

        taps = [
            "depth_head.scratch.refinenet1.out_conv",  # stride ~4
            "depth_head.scratch.refinenet2.out_conv",  # stride ~8
            "depth_head.scratch.refinenet3.out_conv",  # stride ~16
            "depth_head.scratch.refinenet4.out_conv",  # stride ~32
        ]
        modmap = {n: m for n, m in vggt.named_modules()}
        for t in taps:
            if t not in modmap:
                raise KeyError(f"Tap {t} not found in model")
            self.handles.append(modmap[t].register_forward_hook(self._hook(t)))

    def _hook(self, name):
        def fn(m, inp, out):
            self.cache[name] = out
        return fn

    def forward(self, images):
        self.cache.clear()
        _ = self.vggt(images)  # runs full VGGT forward
        return {
            "res2": self.cache["depth_head.scratch.refinenet1.out_conv"],
            "res3": self.cache["depth_head.scratch.refinenet2.out_conv"],
            "res4": self.cache["depth_head.scratch.refinenet3.out_conv"],
            "res5": self.cache["depth_head.scratch.refinenet4.out_conv"],
        }

    def remove_hooks(self):
        for h in self.handles: h.remove()
        self.handles.clear()
