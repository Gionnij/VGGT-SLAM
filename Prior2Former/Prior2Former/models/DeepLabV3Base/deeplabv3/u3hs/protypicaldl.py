from collections import OrderedDict
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mlflow.tracking import MlflowClient

from models.losses import get_loss
from .postprocess import postprocess
from ..panopticdl.base import BaseSegmentationModel

from Visualizations.segmentation import (
    segmentation_to_img,
    get_color_range,
)
from Visualizations.util import (
    plot_embedding_scatter_plot,
    plot_panoptic_prediction,
    scatter_plot,
    tsne_embedding,
)
from torch.nn import functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

# from Stream_Based_AL.utils.import_helper import get_cudf, get_cuml
# import cudf
# import cuml

from .decoder import PrototypicalDeepLabDecoder


class PrototypicalDeepLab(BaseSegmentationModel):
    def __init__(
        self,
        backbone,
        in_channels,
        feature_key,
        low_level_channels,
        low_level_key,
        low_level_channels_project,
        decoder_channels,
        atrous_rates,
        num_classes,
        losses: Dict,
        decoder_args: Dict,
        stuff_classes,
        thing_classes,
        no_class: Dict = {},
        visualize_embedding=True,
        split_rows=None,
        dm=None,
        certainty_stats=False,
        distance_stats=None,
        visualize_prototypes=None,
    ):
        decoder = PrototypicalDeepLabDecoder(
            in_channels,
            feature_key,
            low_level_channels,
            low_level_key,
            low_level_channels_project,
            decoder_channels,
            atrous_rates,
            stuff_classes=stuff_classes,
            thing_classes=thing_classes,
            num_classes=num_classes,
            dm=dm,
            **decoder_args,
        )
        super(PrototypicalDeepLab, self).__init__(backbone, decoder)

        self.losses = {}
        for loss_name in losses:
            self.losses[loss_name] = {
                "weight": losses[loss_name]["weight"],
                "loss": get_loss(**losses[loss_name], num_classes=num_classes, dm=dm),
            }

        self.no_class = no_class

        if "score_init" in no_class and "train" in no_class:
            self.no_class_score = nn.parameter.Parameter(
                torch.tensor([self.no_class["score_init"]]),
                requires_grad=self.no_class["train"],
            )
            self.no_class["no_class_score"] = self.no_class_score

        self.visualize_embedding = visualize_embedding
        self.visualize_prototypes = visualize_prototypes
        self.split_rows = split_rows

        self.certainty_stats = certainty_stats
        if certainty_stats:
            self.register_buffer("certainty_mean", torch.zeros((1)).reshape(()))
            self.register_buffer("certainty_var", torch.ones((1)).reshape(()))

        self.distance_stats = distance_stats
        if distance_stats is not None:
            dim = (num_classes,) if distance_stats.get("per_class", False) else ()
            self.register_buffer("distance_mean", torch.zeros(dim))
            self.register_buffer("distance_var", torch.ones(dim))

        # Initialize parameters.
        self._init_params()

        self.train_metric = []
        self.train_loss = []
        self.val_metric = []
        self.val_loss = []

        self.test_metric = None
        self.test_loss = None

    def set_certainty_stats(self, mean, var):
        self.certainty_mean = mean
        self.certainty_var = var

    def set_distance_stats(self, mean, var):
        self.register_buffer("distance_mean", mean)
        self.register_buffer("distance_var", var)

    def set_semseg_metrics(self, train_stats, val_stats):

        self.train_metric.append(train_stats["metric"])
        self.train_loss.append(train_stats["loss"])
        self.val_metric.append(val_stats["metric"])
        self.val_loss.append(val_stats["loss"])

    def set_semseg_test_metrics(self, test_stats):

        self.test_metric = test_stats["metric"]
        self.test_loss = test_stats["loss"]

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            if isinstance(pred[key], Dict):
                result[key] = self._upsample_predictions(pred[key], input_shape)
            elif len(pred[key].shape) == 4:
                out = F.interpolate(
                    pred[key], size=input_shape, mode="bilinear", align_corners=True
                )
                if "offset" in key:
                    scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                    out *= scale
                result[key] = out
                result[f"{key}_small"] = pred[key]
            else:
                result[key] = pred[key]
        return result

    def loss(self, results, targets, step, epoch, training, dm):
        losses = {"total": 0}

        if "center" in self.losses:
            center_loss_weights = targets["center_weights"][:, None, :, :].expand_as(
                results["detection"]
            )
            center_loss = (
                self.losses["center"]["loss"](results["detection"], targets["center"])
                * center_loss_weights
            )
            # safe division
            if center_loss_weights.sum() > 0:
                center_loss = (
                    center_loss.sum()
                    / center_loss_weights.sum()
                    * self.losses["center"]["weight"]
                )
            else:
                center_loss = center_loss.sum() * 0
            losses["center"] = center_loss
            losses["total"] += center_loss
        if "semantic" in self.losses:
            if "semantic_weights" in targets.keys():
                semantic_loss = (
                    self.losses["semantic"]["loss"](
                        results["semantic"],
                        targets,
                        semantic_weights=targets["semantic_weights"],
                    )
                    * self.losses["semantic"]["weight"]
                )
            else:
                semantic_loss = (
                    self.losses["semantic"]["loss"](
                        results["semantic"], targets, epoch=epoch
                    )
                    * self.losses["semantic"]["weight"]
                )
            losses["semantic"] = semantic_loss
            losses["total"] += semantic_loss
        if "instance" in self.losses:
            if len(self.no_class.keys()) > 0:
                instance_loss = (
                    self.losses["instance"]["loss"](
                        results, targets, no_class_cfg=self.no_class
                    )
                    * self.losses["instance"]["weight"]
                )
            else:
                instance_loss = (
                    self.losses["instance"]["loss"](results, targets)
                    * self.losses["instance"]["weight"]
                )
            losses["instance"] = instance_loss
            losses["total"] += instance_loss
        if (
            "discriminative" in self.losses
            and self.losses["discriminative"]["weight"] > 0
        ):
            discriminative_loss = (
                self.losses["discriminative"]["loss"](
                    results["embedding"], targets, num_classes=dm.num_classes
                )["total"]
                * self.losses["discriminative"]["weight"]
            )
            losses["discriminative"] = discriminative_loss
            losses["total"] += discriminative_loss
        if "rgb" in self.losses:
            rgb_loss = (
                self.losses["rgb"]["loss"](results["rgb"]["result"], targets["image"])
                * self.losses["rgb"]["weight"]
            )
            losses["rgb"] = rgb_loss.sum() * self.losses["rgb"]["weight"]
            losses["total"] += rgb_loss.sum() * self.losses["rgb"]["weight"]
        if self.no_class is not None and len(self.no_class.keys()) > 0:
            losses["no_class"] = self.no_class["no_class_score"]

        return losses

    def postprocess(self, out, dm, post_process_conf):
        extra_args = {}
        if self.certainty_stats:
            extra_args = {
                "certainty_mean": self.certainty_mean,
                "certainty_var": self.certainty_var,
            }
        if self.distance_stats is not None:
            extra_args["distance_mean"] = self.distance_mean
            extra_args["distance_var"] = self.distance_var

        return postprocess(
            out,
            dm=dm,
            no_class_cfg=self.no_class if len(self.no_class.keys()) > 0 else None,
            semantic_head=self.decoder.semantic_head,
            **post_process_conf,
            **extra_args,
        )

    def training_step(self, batch, out):
        self.decoder.training_step(batch, out)

    def plot_prediction(
        self,
        x,
        batch,
        postprocess,
        out,
        dm,
        experiment,
        batch_idx,
        global_step,
        log_folder,
        oodis=False,
        ignore_instance=None,
    ):
        detections = x * torch.tensor(dm.std, device=x.device).reshape(
            (3, 1, 1)
        ) + torch.tensor(dm.mean, device=x.device).reshape((3, 1, 1))
        colors = [col.to(x.device) for col in dm.get_class_colors()]
        dets = torch.clip(out["detection"], min=0, max=1).to(
            x.device
        )  # object center prediction

        detection_per_class = dets.shape[1] == len(dm.mapped_thing_list)

        if detection_per_class:
            for i, thing_id in enumerate(dm.mapped_thing_list):
                det = dets[:, i : i + 1]
                color = colors[thing_id].unsqueeze(0).unsqueeze(2).unsqueeze(2)
                detections = (1 - det) * detections + det * color
        else:
            color = torch.tensor([1.0, 0, 0], device=x.device).reshape((1, 3, 1, 1))
            # detections = (1 - dets[:, 0]) * detections + dets[:, 0] * color
            detections = (1 - dets) * detections + dets * color

        detections_post = x * torch.tensor(dm.std, device=x.device).reshape(
            (3, 1, 1)
        ) + torch.tensor(dm.mean, device=x.device).reshape((3, 1, 1))
        batch_size, height, width = (
            detections_post.shape[0],
            detections_post.shape[2],
            detections_post.shape[3],
        )
        centers = postprocess["instance_ctrs"]

        positions = torch.stack(
            (
                torch.arange(0, height).reshape((height, 1)).expand((height, width)),
                torch.arange(0, width).reshape((1, width)).expand((height, width)),
            )
        )

        distances = torch.zeros_like(detections_post)
        uncertainties = torch.zeros_like(detections_post)

        for i in range(batch_size):
            batch_centers = centers[i].cpu()

            uncertainties[i, 0] = 1 - postprocess["certainties"][i]
            uncertainties[i, 0] = (
                uncertainties[i, 0] - uncertainties[i, 0].min()
            ) / uncertainties[i, 0].max()

            dists = postprocess["distances"][i]
            distances[i, 0] = (
                dists / dists[dm.get_semantic_from_batch(batch)[i] != 255].max()
            ).clip(0, 1)

            prototype_classes = postprocess["prototype_classes"][i].cpu()

            for j, ctr in enumerate(batch_centers):
                clss, y_pos, x_pos = ctr[0].item(), ctr[1].item(), ctr[2].item()
                if detection_per_class:
                    clss = dm.mapped_thing_list[clss]
                else:
                    clss = prototype_classes[j].item()
                mask = ((positions[0] - y_pos) ** 2 + (positions[1] - x_pos) ** 2) < 25
                detections_post[i, :, mask] = colors[int(clss)].unsqueeze(1)

        extra_vis = [
            detections.cpu(),
            detections_post.cpu(),
            uncertainties.cpu(),
            distances.cpu(),
        ]
        if "rgb" in out:
            out_rgb = out["rgb"]["result"]
            image = out_rgb.cpu().detach()
            image = dm.unnormalize(image)
            extra_vis.append(image)
        if "ood" in batch:
            ood_batch = x * torch.tensor(dm.std, device=x.device).reshape(
                (3, 1, 1)
            ) + torch.tensor(dm.mean, device=x.device).reshape((3, 1, 1))
            ood_batch = ood_batch.permute((1, 0, 2, 3))

            ood_mask = batch["ood"] == 1

            ood_batch[:, ood_mask] = ood_batch[:, ood_mask] * 0.5 + 0.5 * torch.tensor(
                [1.0, 0.0, 0.0], device=x.device
            ).reshape((3, 1))
            extra_vis.append(ood_batch.permute(1, 0, 2, 3).cpu())

        if self.visualize_embedding:
            tsne_embedding = self.visualize_embed(out, postprocess)
            extra_vis.append(tsne_embedding)
        if "ood_mask" in postprocess:
            ood_mask = postprocess["ood_mask"]
            ood_mask = ood_mask.int()
            outlier_flag = (ood_mask == -1).any()
            ood_mask[ood_mask == -1] = ood_mask.max() + 1
            colors = get_color_range(ood_mask.max() + 1 - outlier_flag.int())
            if len(colors) < 2 and outlier_flag:
                colors = [torch.zeros(3), torch.ones(3)]
            elif outlier_flag:
                colors[0] = torch.zeros(3)
                colors.append(torch.ones(3))
            else:
                colors[0] = torch.zeros(3)
            if len(ood_mask.shape) == 2:
                ood_mask = ood_mask.unsqueeze(0)
            ood_mask_img = segmentation_to_img(
                ood_mask,
                ood_mask.max() + 1,
                colors=colors,
            ).squeeze()
            extra_vis.append(ood_mask_img)
        if "entropy" in postprocess:
            entropy = torch.zeros_like(detections_post)
            for b in range(batch_size):
                entropy[b, 0] = postprocess["entropy"][b] / 255
            extra_vis.append(entropy)

        if "logit_entropy" in postprocess:

            entropy = torch.zeros_like(detections_post)
            for b in range(batch_size):
                entropy[b, 0] = postprocess["logit_entropy"][b]
            extra_vis.append(entropy)

        grid, panoptic_gt_img, _ = plot_panoptic_prediction(
            x,
            dm.get_semantic_from_batch(batch),
            batch["instance"],
            postprocess["semantic"],
            postprocess["panoptic"],
            datamodule=dm,
            extra_imgs=extra_vis,
            split_rows=self.split_rows,
            ignore_instance=ignore_instance,
            return_list=oodis,
        )
        if experiment is None:
            return grid, panoptic_gt_img
        if isinstance(experiment, MlflowClient):
            img = Image.fromarray(
                (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            img.save(os.path.join(log_folder, f"current/predictions-{batch_idx}.png"))
            # experiment.log_image(
            #     run_id=self.run_id,
            #     image=img,
            #     artifact_file=f"current/predictions-{batch_idx}.png",
            # )
        else:
            experiment.add_image(f"predictions-{batch_idx}", grid, global_step)

        if self.visualize_embedding:
            for i in range(tsne_embedding.shape[0]):
                colors = panoptic_gt_img[i].reshape((3, -1))
                embedding = tsne_embedding[i].contiguous().reshape((3, -1))

                if not self.decoder.extra_thing_prototypes:
                    ctrs = postprocess["instance_ctrs"][i][:, 1:]
                    height, width = tsne_embedding.shape[2], tsne_embedding.shape[3]

                    sizes = torch.ones_like(tsne_embedding[0, 0]).reshape((-1))
                    zorders = torch.ones((height * width), dtype=torch.int32)

                    indices = ctrs[:, 1] + ctrs[:, 0] * width

                    sizes[indices] = 100
                    zorders[indices] = 2

                    colors[:, indices] = colors[:, indices] * 0.8 + torch.tensor(
                        [0.2, 0, 0]
                    ).reshape((3, 1))

                    sorted_z, ind = zorders.sort()

                    sizes = sizes.gather(dim=0, index=ind)
                    colors = colors.gather(
                        dim=1, index=ind.unsqueeze(0).expand((3, ind.shape[0]))
                    )
                    embedding = embedding.gather(
                        dim=1,
                        index=ind.unsqueeze(0).expand((3, ind.shape[0])),
                    )

                    sizes = sizes.cpu().numpy()
                else:
                    sizes = 1

                scatter_plot = plot_embedding_scatter_plot(
                    embedding.unsqueeze(2), colors, sizes=sizes
                )

                if isinstance(experiment, MlflowClient):
                    img = Image.fromarray(
                        (scatter_plot.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    )
                    img.save(
                        os.path.join(log_folder, f"current/embeddings-{batch_idx}.png")
                    )
                    # experiment.log_image(
                    #     run_id=self.run_id,
                    #     image=img,
                    #     artifact_file=f"current/embeddings-{batch_idx}.png",
                    # )
                else:
                    experiment.add_image(
                        f"embeddings-{batch_idx}-{i}", scatter_plot, global_step
                    )

    def collect_set_data(self, batch, out, postprocess, dm):
        if self.visualize_prototypes is not None:
            if (
                self.visualize_prototypes.get("embedding_source", "prototypes")
                == "prototypes"
            ):
                return postprocess["prototypes"], postprocess["prototype_classes"]
            elif (
                self.visualize_prototypes.get("embedding_source", "prototypes")
                == "mean_embedding"
            ):
                embeds = out["embedding"]
                panoptic = postprocess["panoptic"]

                batch_size = embeds.shape[0]

                mean_embeds = []
                clsses = []

                for i in range(batch_size):
                    remapped = torch.zeros_like(panoptic[i])

                    unique = panoptic[i].unique()
                    clss = unique // dm.label_divisor

                    for j, x in enumerate(unique):
                        remapped[panoptic[i] == x] = j

                    oh = F.one_hot(remapped)
                    counts = oh.sum(dim=(0, 1))

                    sums = (oh.unsqueeze(0) * embeds[i].unsqueeze(3)).sum(dim=(1, 2))
                    means = sums / counts.unsqueeze(0)

                    mean_embeds.append(means.T)
                    clsses.append(clss)

                return mean_embeds, clsses

    def process_set_data(self, valset_data, experiment, dm):
        if self.visualize_prototypes is not None:
            prototypes = None
            prototype_classes = None
            frame_ids = None
            i = 0
            for batch_prototypes, batch_classes in valset_data:
                for prot, prot_class in zip(batch_prototypes, batch_classes):
                    prot_class = prot_class[: prot.shape[0]]
                    if prototypes is None:
                        prototypes = prot
                        prototype_classes = prot_class
                        frame_ids = torch.ones_like(prot_class) * i
                    else:
                        prototypes = torch.cat((prototypes, prot), dim=0)
                        prototype_classes = torch.cat(
                            (prototype_classes, prot_class), dim=0
                        )
                        frame_ids = torch.cat(
                            (frame_ids, torch.ones_like(prot_class) * i), dim=0
                        )
                    i = i + 1

            # cudf = get_cudf()
            # cuml = get_cuml()
            import cuml
            import cudf

            data = prototypes[:, : self.decoder.feature_dim]
            df = cudf.from_dlpack(to_dlpack(data))

            if self.visualize_prototypes.get("embedder", "TSNE") == "TSNE":
                tsne = cuml.TSNE(
                    n_components=2,
                    n_iter=2000,
                    **self.visualize_prototypes.get("embedder_args", {}),
                )
                embedded = tsne.fit_transform(df)

                embedded = from_dlpack(embedded.to_dlpack()).contiguous()
            elif self.visualize_prototypes.get("embedder", "TSNE") == "PCA":
                pca = cuml.PCA(
                    n_components=2, **self.visualize_prototypes.get("embedder_args", {})
                )
                embedded = pca.fit_transform(df)

                embedded = from_dlpack(embedded.to_dlpack()).contiguous()
            else:
                raise Exception(
                    f"Embedder called {self.visualize_prototypes.get('embedder')} not known"
                )

            class_colors = torch.stack(dm.get_class_colors()).to(embedded.device)

            if (
                self.visualize_prototypes.get("color_by", "predicted_class")
                == "predicted_class"
            ):
                point_colors = class_colors.gather(
                    dim=0,
                    index=prototype_classes.unsqueeze(1).expand(
                        (prototype_classes.shape[0], 3)
                    ),
                )
            elif (
                self.visualize_prototypes.get("color_by", "predicted_class")
                == "frame_id"
            ):
                index = frame_ids % class_colors.shape[0]
                point_colors = class_colors.gather(
                    dim=0,
                    index=index.unsqueeze(1).expand((prototype_classes.shape[0], 3)),
                )
            else:
                raise Exception(
                    f"Coloring called {self.visualize_prototypes.get('color_by')} not known"
                )

            scatter = scatter_plot(embedded.cpu().numpy(), point_colors.cpu().numpy())
            experiment.add_image(f"prototype-embeddings", scatter)

    def visualize_embed(self, out, postprocess):
        tsne_embed = tsne_embedding(out["embedding_small"])

        tsne_embed = F.interpolate(
            tsne_embed,
            size=out["embedding"].shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        return tsne_embed

    def set_output_stride(self, os):
        self.backbone.set_output_stride(os)
        self.decoder.set_output_stride(os)
