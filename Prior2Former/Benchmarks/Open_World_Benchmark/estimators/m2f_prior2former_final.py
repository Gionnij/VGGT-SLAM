import torch
import torch.nn.functional as F



class _m2f_beta:
    def __init__(
        self, out_size=None, refinement_mask=False, use_semseg=False, **kwargs
    ):
        self.refinement_mask = refinement_mask

        self.out_size = out_size
        self.use_semseg = use_semseg
        self.sml = kwargs.get("sml", False)
        self.normalize = kwargs.get("normalize", True)
        self.temperature = kwargs.get("temperature", 1)
        self.classificationLogits = kwargs.get("classificationLogits", True)
        self.non_object_class=kwargs.get("non_object_class", True)
        self.threshold = kwargs.get("threshold", 0.0)
        self.kt = kwargs.get("keep", 0)

    def set_mean_std(self, mean=None, std=None):
        return None

    def __call__(self, model, x=None, out=None):
        out = out if out is not None else model(x)

        if self.classificationLogits or self.sml == "mask_cls":
            logits = out["pred_logits"].squeeze()[..., :-1]
        elif not self.non_object_class:
            logits = model.sem_seg_head.predictor.class_embed.probabilities(
                out["pred_logits"]
            ).squeeze()
        else:
            logits = model.sem_seg_head.predictor.class_embed.probabilities(
                out["pred_logits"]
            ).squeeze()[..., :-1]
        alpha = out["pred_masks"][:, 0]
        beta = out["pred_masks"][:, 1]

        if x is not None or self.out_size is not None:
            alpha = F.interpolate(
                alpha,
                size=x.shape[-2:] if self.out_size is None else self.out_size,
                mode="bilinear",
                # align_corners=False,
            ).squeeze()
            beta = F.interpolate(
                beta,
                size=x.shape[-2:] if self.out_size is None else self.out_size,
                mode="bilinear",
                # align_corners=False,
            ).squeeze()
        if self.use_semseg:
            mask_pred = alpha / (alpha + beta)
            semseg = (
                torch.einsum("qc,qhw->chw", logits, mask_pred.squeeze()).max(0).values
            )

        else:
            semseg = 1
        return alpha.squeeze(), beta.squeeze(), logits, semseg


class _m2f_beta_keep(_m2f_beta):

    def __call__(self, model, x=None, out=None):
        out = out if out is not None else model(x)
        scores, labels = model.sem_seg_head.predictor.class_embed.probabilities(
            out["pred_logits"]
        ).max(-1)
        if self.kt == 0:
            keep = labels.ne(model.sem_seg_head.num_classes) & (
                scores
                > self.threshold  # model.object_mask_threshold  # softmax threshholding
            )
        elif self.kt == 2:
            keep = ~((labels == model.sem_seg_head.num_classes) & (scores > 0.90))
        else:
            raise NotImplementedError()
        if keep.sum() <= 1:
            return None, None, None, None
        #print(model.sem_seg_head.predictor.__class__.__name__, model.sem_seg_head.predictor.class_embed.__class__.__name__)
        if model.sem_seg_head.predictor.__class__.__name__ == "MultiScaleMaskedTransformerDecoder_PN2":
            # In this case we need a specific filtering since there is no no-object class
            keep = model.sem_seg_head.predictor.class_embed.probabilities(out["pred_logits"]).max(-1).values > 0.3
        elif model.sem_seg_head.predictor.class_embed.__class__.__name__ ==  "DPNHead_BinaryNNO":
            keep = model.sem_seg_head.predictor.class_embed.mask_probabilities(out["pred_logits"]) > 0.5


        alpha = out["pred_masks"][:, 0]
        beta = out["pred_masks"][:, 1]
        if self.classificationLogits or self.sml == "mask_cls":
            scores = out["pred_logits"][..., :-1][keep]
        else:
            scores = model.sem_seg_head.predictor.class_embed.probabilities(
                out["pred_logits"]
            )[..., :-1][keep]
        if x is not None or self.out_size is not None:
            alpha = F.interpolate(
                alpha,
                size=x.shape[-2:] if self.out_size is None else self.out_size,
                mode="bilinear",
                # align_corners=False,
            )
            beta = F.interpolate(
                beta,
                size=x.shape[-2:] if self.out_size is None else self.out_size,
                mode="bilinear",
                # align_corners=False,
            )
        if self.use_semseg:
            # use full model for semantic predicion and keep only for mask prediction
            logits = model.sem_seg_head.predictor.class_embed.probabilities(
                out["pred_logits"]
            ).squeeze()[..., :-1]
            mask_pred = alpha / (alpha + beta)
            semseg = (
                torch.einsum("qc,qhw->chw", logits, mask_pred.squeeze()).max(0).values
            )

        else:
            semseg = 1
        alpha = alpha[keep]
        beta = beta[keep]
        return alpha, beta, scores, semseg


class max_alpha_beta_cls_uncertainty(_m2f_beta):
    """Taking the mask with the max alpha score and multiplying 
    the classification probability with the mask probability.

    Args:
        _m2f_beta (_type_): _description_
    """
    def __init__(
        self, out_size=None, refinement_mask=False, use_semseg=False, **kwargs
    ):
        super().__init__(out_size, refinement_mask, use_semseg, **kwargs)
        self.classificationLogits = False

    def __call__(self, model, x=None, out=None):
        alpha, beta, mask_cls, _ = super().__call__(model, x, out)
        if alpha is None:
            return (
                torch.zeros(x.shape[-2:])
                if self.out_size is None
                else torch.zeros(self.out_size)
            )
        mm = (alpha.squeeze()).argmax(0)

        alpha_max = (alpha.squeeze()).max(0)[0]

        beta_s = beta.squeeze()

        beta_max = torch.gather(beta_s, 0, mm.unsqueeze(0).expand(beta_s.shape)).mean(0)

        tt = mask_cls.squeeze().sum(1)

        uncertainty = -(tt[mm] * ((alpha_max / (alpha_max + beta_max))))
        if self.normalize:
            uncertainty = (uncertainty - uncertainty.min()) / (
                uncertainty.max() - uncertainty.min()
            )
        return uncertainty


class max_alpha_beta_cls_uncertainty_keep(_m2f_beta_keep):
    """Final method of P2F, taking the mask with the max alpha score and multiplying 
    the classification probability with the mask probability.

    Args:
        _m2f_beta_keep (_type_): _description_
    """

    def __init__(
        self, out_size=None, refinement_mask=False, use_semseg=False, **kwargs
    ):
        super().__init__(out_size, refinement_mask, use_semseg, **kwargs)
        self.classificationLogits = False

    def __call__(self, model, x=None, out=None):
        alpha, beta, mask_cls, _ = super().__call__(model, x, out)
        if alpha is None:
            return (
                torch.zeros(x.shape[-2:])
                if self.out_size is None
                else torch.zeros(self.out_size)
            )
        mm = (alpha.squeeze()).argmax(0)

        alpha_max = (alpha.squeeze()).max(0)[0]

        beta_s = beta.squeeze()

        beta_max = torch.gather(beta_s, 0, mm.unsqueeze(0).expand(beta_s.shape)).mean(0)

        tt = mask_cls.squeeze().sum(1)

        uncertainty = -(tt[mm] * ((alpha_max / (alpha_max + beta_max))))
        if self.normalize:
            uncertainty = (uncertainty - uncertainty.min()) / (
                uncertainty.max() - uncertainty.min()
            )
        return uncertainty