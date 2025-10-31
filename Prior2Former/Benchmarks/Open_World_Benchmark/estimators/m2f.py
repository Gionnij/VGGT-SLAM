import torch
import torch.nn.functional as F


class _m2f:

    def __init__(
        self, out_size=None, refinement_mask=False, use_semseg=False, **kwargs
    ):
        self.refinement_mask = refinement_mask
        self.out_size = out_size
        self.normalize = kwargs.get("normalize", True)
        self.classificationLogits = kwargs.get("classificationLogits", True)
        self.temperature = kwargs.get("temperature", 1)

    def __call__(self, model, x=None, out=None):
        out = out if out is not None else model(x)
        if self.classificationLogits:
            logits = out["pred_logits"].squeeze()[..., :-1]
        else:
            logits = model.sem_seg_head.predictor.class_embed.probabilities(
                out["pred_logits"]
            ).squeeze()[..., :-1]
        masks = out["pred_masks"]
        if x is not None or self.out_size is not None:
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(0)
            masks = (
                F.interpolate(
                    masks,
                    size=x.shape[-2:] if self.out_size is None else self.out_size,
                    mode="bilinear",
                    # align_corners=False,
                )
                .squeeze()
                .sigmoid()
            )
        return masks, logits
    

    def set_mean_std(self, mean=None, std=None):
        return None


class _m2f_keep:

    def __init__(
        self, out_size=None, refinement_mask=False, use_semseg=False, **kwargs
    ):
        self.refinement_mask = refinement_mask
        self.out_size = out_size
        self.normalize = kwargs.get("normalize", True)
        self.classificationLogits = kwargs.get("classificationLogits", True)
        self.temperature = kwargs.get("temperature", 1)

    def __call__(self, model, x=None, out=None):
        out = out if out is not None else model(x)
        scores, labels = model.sem_seg_head.predictor.class_embed.probabilities(
            out["pred_logits"]
        ).max(-1)
        keep = labels.ne(model.sem_seg_head.num_classes) & (
            scores > 0  # model.object_mask_threshold  # softmax threshholding
        )
        if keep.sum() <= 1:
            return None, None
        if self.classificationLogits:
            logits = out["pred_logits"][..., :-1][keep]
        else:
            logits = model.sem_seg_head.predictor.class_embed.probabilities(
                out["pred_logits"]
            )[..., :-1][keep]
        masks = out["pred_masks"][keep]
        if x is not None or self.out_size is not None:
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(0)
            masks = (
                F.interpolate(
                    masks,
                    size=x.shape[-2:] if self.out_size is None else self.out_size,
                    mode="bilinear",
                    # align_corners=False,
                )
                .squeeze()
                .sigmoid()
            )
        return masks, logits



class m2f_logit_uncertainty(_m2f):

    def __call__(self, model, x=None, out=None):
        masks, logits = super().__call__(model, x, out)
        semseg = torch.einsum("qc,qhw->chw", logits, masks.squeeze())
        semseg /= self.temperature
        confidence = semseg.cpu().detach().max(0).values
        if self.normalize:
            confidence = (confidence - confidence.min()) / (
                confidence.max() - confidence.min()
            )
        if self.refinement_mask:
            road_id = model.dm.from_cityscapes_id[7]  # road id
            stuff_ids = max(model.dm.mapped_stuff_list)
            semantic = semseg.argmax(0)
            r_mask = (semantic.squeeze() <= stuff_ids) * (semantic.squeeze() != road_id)
            confidence[r_mask] = 1
        return 1 - confidence.squeeze()


class m2f_logit_uncertainty_keep(_m2f_keep):

    def __call__(self, model, x=None, out=None):
        masks, logits = super().__call__(model, x, out)
        if masks is None:
            return (
                torch.zeros(x.shape[-2:])
                if self.out_size is None
                else torch.zeros(self.out_size)
            )
        semseg = torch.einsum("qc,qhw->chw", logits, masks.squeeze())
        confidence = semseg.cpu().detach().max(0).values
        if self.normalize:
            confidence = (confidence - confidence.min()) / (
                confidence.max() - confidence.min()
            )
        if self.refinement_mask:
            road_id = model.dm.from_cityscapes_id[7]  # road id
            stuff_ids = max(model.dm.mapped_stuff_list)
            semantic = semseg.argmax(0)
            r_mask = (semantic.squeeze() <= stuff_ids) * (semantic.squeeze() != road_id)
            confidence[r_mask] = 1
        return 1 - confidence.squeeze()


class m2f_mask_max(_m2f):

    def __call__(self, model, x=None, out=None):
        masks, logits = super().__call__(model, x, out)
        certainty = masks.max(0).values
        if self.normalize:
            certainty = (certainty - certainty.min()) / (
                certainty.max() - certainty.min()
            )
        if self.refinement_mask:
            road_id = model.dm.from_cityscapes_id[7]  # road id
            stuff_ids = max(model.dm.mapped_stuff_list)
            semantic = model.semantic_inference(
                out["pred_logits"], out["pred_masks"]
            ).argmax(0)
            r_mask = (semantic.squeeze() <= stuff_ids) * (semantic.squeeze() != road_id)
            certainty[r_mask] = 1
        return 1 - certainty


class m2f_mask_max_keep(_m2f_keep):

    def __call__(self, model, x=None, out=None):
        masks, logits = super().__call__(model, x, out)
        if masks is None:
            return (
                torch.zeros(x.shape[-2:])
                if self.out_size is None
                else torch.zeros(self.out_size)
            )
        certainty = masks.max(0).values
        if self.normalize:
            certainty = (certainty - certainty.min()) / (
                certainty.max() - certainty.min()
            )
        if self.refinement_mask:
            road_id = model.dm.from_cityscapes_id[7]  # road id
            stuff_ids = max(model.dm.mapped_stuff_list)
            semantic = model.semantic_inference(
                out["pred_logits"], out["pred_masks"]
            ).argmax(0)
            r_mask = (semantic.squeeze() <= stuff_ids) * (semantic.squeeze() != road_id)
            certainty[r_mask] = 1
        return 1 - certainty

class mask2anomaly(_m2f):

    def __init__(self, out_size=None, refinement_mask=False, **kwargs):
        self.refinement_mask = refinement_mask
        self.out_size = out_size
        self.normalize = kwargs.get("normalize", True)

    def semantic_inference(self, model, mask_cls, mask_pred):

        mask_cls_f = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred_f = mask_pred.clone()
        semseg = torch.einsum("qc,qhw->chw", mask_cls_f, mask_pred_f)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.clone()
        keep = (
            labels.ne(model.sem_seg_head.num_classes)
            & (scores > 0.95)
            & (labels < 11)
            & (labels > 1)
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        semseg = torch.cat((semseg, cur_prob_masks), 0)
        print("keep: ", keep.sum())
        return semseg

    def __call__(self, model, x=None, out=None):
        out = out if out is not None else model(x)
        logits = out["pred_logits"].squeeze()
        masks = out["pred_masks"]
        if x is not None or self.out_size is not None:
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(0)
            masks = (
                F.interpolate(
                    masks,
                    size=x.shape[-2:] if self.out_size is None else self.out_size,
                    mode="bilinear",
                    # align_corners=False,
                )
                .squeeze()
                .sigmoid()
            )
        semseg = self.semantic_inference(model, logits, masks)
        outputs_na = 1 - torch.max(semseg[0:19, :, :].unsqueeze(0), axis=1)[0]
        if semseg[19:, :, :].shape[0] > 1:
            outputs_na_mask = torch.max(semseg[19:, :, :].unsqueeze(0), axis=1)[0]
            outputs_na_mask[outputs_na_mask < 0.5] = 0
            outputs_na_mask[outputs_na_mask >= 0.5] = 1
            outputs_na_mask = 1 - outputs_na_mask
            outputs_na = (outputs_na * outputs_na_mask.detach()).squeeze().squeeze()
            if self.normalize:
                outputs_na = (outputs_na - outputs_na.min()) / (
                    outputs_na.max() - outputs_na.min()
                )
            return outputs_na
        else:
            return torch.zeros(semseg.shape[-2:])


class rba(_m2f):
    """Rejected by all (RBA) uncertainty measure. M2F baseline from paper

    Args:
        _m2f (_type_): _description_
    """
    def __init__(self, out_size=None, refinement_mask=False, **kwargs):
        super().__init__(out_size, refinement_mask, classificationLogits=False)

    def __call__(self, model, x=None, out=None):
        masks, logits = super().__call__(model, x, out)
        semseg = -torch.einsum("qc,qhw->chw", logits, masks.squeeze()).tanh().sum(0)
        if self.normalize:
            semseg = (semseg - semseg.min()) / (semseg.max() - semseg.min())
        return semseg
