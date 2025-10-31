from .m2f import (
    m2f_logit_uncertainty,
    m2f_logit_uncertainty_keep,
    mask2anomaly,
    rba,
    m2f_mask_max,
    m2f_mask_max_keep
)

from .m2f_prior2former_final import (
    max_alpha_beta_cls_uncertainty,
    max_alpha_beta_cls_uncertainty_keep,
)


def get_estimator(name, args):
    if name == "rba":
        return rba(**args)
    elif name == "m2f_mask_max":
        return m2f_mask_max(**args)
    elif name == "m2f_mask_max_keep":
        return m2f_mask_max_keep(**args)
    elif name == "max_alpha_beta_cls_uncertainty":
        return max_alpha_beta_cls_uncertainty(**args)
    elif name == "max_alpha_beta_cls_uncertainty_keep": #Prior2Former
        return max_alpha_beta_cls_uncertainty_keep(**args)
    elif name == "max_alpha_beta_cls_uncertainty":
        return max_alpha_beta_cls_uncertainty(**args)
    elif name == "mask2anomaly":
        return mask2anomaly(**args)
    elif name == "m2f_logit_uncertainty_keep":
        return m2f_logit_uncertainty_keep(**args)
    elif name == "m2f_logit_uncertainty":
        return m2f_logit_uncertainty(**args)
    else:
        raise NotImplementedError(f"Estimator of name {name} is not known")
