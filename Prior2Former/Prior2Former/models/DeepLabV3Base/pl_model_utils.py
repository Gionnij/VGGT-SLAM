from typing import Dict


def get_model(args: Dict, dm, **kwargs):
    if args["name"] == "u3hs": #U3HS
        from .deeplabv3.u3hs import u3hs

        return u3hs(**args, dm=dm)
    elif args["name"] == "mask2former":
        from models.mask2former import MaskFormer

        return MaskFormer(**MaskFormer.from_config(args, dm=dm), metadata=None, dm=dm)
    else:
        raise NotImplementedError("")

