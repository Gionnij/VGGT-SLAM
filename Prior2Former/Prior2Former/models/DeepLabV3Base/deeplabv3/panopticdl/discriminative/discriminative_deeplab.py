# ------------------------------------------------------------------------------
# Panoptic-DeepLab meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from typing import Dict

import torch
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    remap_instance,
)
from models.losses import get_loss
from ..base import BaseSegmentationModel
from Visualizations.util import (
    plot_embedding_scatter_plot,
    plot_panoptic_prediction,
    tsne_embedding,
)
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from torch.nn import functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

# from Stream_Based_AL.utils.import_helper import get_cudf, get_cuml
# import cudf
# import cuml
from .decoder import DiscriminativeDeepLabDecoder


class DiscriminativeDeepLab(BaseSegmentationModel):
    """
    Implements Discriminative-DeepLab, which combines the architecture of panoptic deeplab with
    the instance segmentation approach from "Semantic Instance Segmentation with a Discriminative Loss Function" (https://arxiv.org/pdf/1708.02551.pdf)
    """

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
        **kwargs,
    ):
        decoder = DiscriminativeDeepLabDecoder(
            in_channels,
            feature_key,
            low_level_channels,
            low_level_key,
            low_level_channels_project,
            decoder_channels,
            atrous_rates,
            num_classes,
            **kwargs,
        )
        super(DiscriminativeDeepLab, self).__init__(backbone, decoder)

        self.num_classes = num_classes

        self.losses = {}
        for loss_name in losses:
            self.losses[loss_name] = {
                "loss": get_loss(**losses[loss_name]),
                **losses[loss_name],
            }

        # Initialize parameters.
        self._init_params()

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
            out = F.interpolate(
                pred[key], size=input_shape, mode="bilinear", align_corners=True
            )
            if "offset" in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
            result[f"{key}_small"] = pred[key]
        return result

    def loss(self, results, targets=None, step=0, epoch=0, training=True, dm=None):
        losses = {"total": 0}
        if targets is not None:
            if "semantic_weights" in targets.keys():
                semantic_loss = (
                    self.losses["semantic"]["loss"](
                        results["semantic"],
                        targets["semantic"],
                        semantic_weights=targets["semantic_weights"],
                    )
                    * self.losses["semantic"]["weight"]
                )
            else:
                semantic_loss = (
                    self.losses["semantic"]["loss"](
                        results["semantic"], targets["semantic"]
                    )
                    * self.losses["semantic"]["weight"]
                )
            losses["semantic"] = semantic_loss
            losses["total"] += semantic_loss

            if (
                "after_epochs" not in self.losses["instance"]
                or epoch >= self.losses["instance"]["after_epochs"]
            ):
                instance_loss = self.losses["instance"]["loss"](
                    results["embedding"],
                    targets,
                    out=results,
                    num_classes=self.num_classes,
                    dm=dm,
                )

                losses["instance"] = instance_loss["total"]
                for k in instance_loss.keys():
                    if k != "total":
                        losses[f"instance/{k}"] = instance_loss[k]

                losses["total"] += (
                    losses["instance"] * self.losses["instance"]["weight"]
                )
        return losses

    def postprocess(self, out, dm, post_process_conf):
        batch_size = out["embedding"].shape[0]

        semantic_preds = []
        panoptic_preds = []

        # cudf = get_cudf()
        # cuml = get_cuml()
        import cuml
        import cudf

        for i in range(batch_size):
            panoptic_id, semantic_pred = self.postprocess_frame(
                out, post_process_conf, i, dm, cuml, cudf
            )

            panoptic_preds.append(panoptic_id)
            semantic_preds.append(semantic_pred)

        semantic_preds = torch.stack(semantic_preds).to(out["embedding"].device)
        panoptic_preds = torch.stack(panoptic_preds).to(out["embedding"].device)

        return {
            "semantic": semantic_preds,
            "panoptic": panoptic_preds,
        }

    def postprocess_frame(self, out, post_process_conf, i, dm, cuml, cudf):
        embedding = out["embedding"][i].cpu().detach().numpy()
        if post_process_conf.get("small", False):
            embedding = out["embedding_small"][i].cpu().detach().numpy()

        embedding_reshaped = embedding.reshape((embedding.shape[0], -1)).T

        instance_ids = self.cluster(
            post_process_conf, embedding_reshaped, dm, cuml, cudf, out, i
        )

        instance_ids[instance_ids == -1] = instance_ids.max() + 1

        semantic_pred = out["semantic"][i].argmax(dim=0)
        instance_ids = instance_ids.reshape(embedding.shape[1:])
        if (
            post_process_conf.get("small", False)
            and instance_ids.shape[0] < semantic_pred.shape[0]
        ):
            instance_ids = (
                F.interpolate(
                    instance_ids.unsqueeze(0).unsqueeze(0).float(),
                    size=semantic_pred.shape,
                    mode="nearest",
                )
                .squeeze()
                .long()
            )
        # This is done since during the first few epochs DBSCAN might find a lot more clusters than actually exist
        instance_ids = instance_ids % dm.label_divisor
        instance_ids = remap_instance(instance_ids, -1)

        if post_process_conf.get("majority_vote", False):
            panoptic_id = self.compute_majority_vote(
                instance_ids, semantic_pred, out, i, dm
            )
        else:
            thing_set = set(dm.mapped_thing_list)
            for i in semantic_pred.unique():
                i = i.item()
                if i not in thing_set:
                    instance_ids[semantic_pred == i] = 0

            panoptic_id = semantic_pred * dm.label_divisor + instance_ids

        return panoptic_id, semantic_pred

    def cluster(self, post_process_conf, embedding_reshaped, dm, cuml, cudf, out, i):
        if post_process_conf.get("per_class", False):
            return self.cluster_per_class(post_process_conf, out, i, dm, cuml, cudf)

        clusterer = self.get_clusterer(post_process_conf, cuml, dm)
        if post_process_conf.get("type", "dbscan") == "dbscan_gpu":
            data = out[
                (
                    "embedding_small"
                    if post_process_conf.get("small", False)
                    else "embedding"
                )
            ][i].detach()
            data = data.reshape((data.shape[0], -1)).T
            df = cudf.from_dlpack(to_dlpack(data))

            clusterer.fit(df)
            instance_ids = from_dlpack(clusterer.labels_.to_dlpack())
            instance_ids = remap_instance(instance_ids, -1)

            if post_process_conf.get("refit", False):
                embedding = out["embedding"][i].detach()
                embeds_small = out["embedding_small"][i].detach()

                unique, counts = instance_ids.unique(return_counts=True)
                n_instances = unique.shape[0] - 1

                if n_instances > 0:
                    embed_dim = embedding.shape[0]

                    instance_ids[instance_ids == -1] = n_instances
                    instance_one_hot = F.one_hot(
                        instance_ids.to(torch.int64), n_instances + 1
                    )[:, :n_instances]

                    instance_means = torch.einsum(
                        "en,nc->ec",
                        embeds_small.reshape((embed_dim, -1)),
                        instance_one_hot.float(),
                    ) / counts[unique != -1].unsqueeze(0)

                    distances = torch.cdist(
                        instance_means.transpose(0, 1).unsqueeze(0),
                        embedding.reshape((embed_dim, -1)).transpose(0, 1).unsqueeze(0),
                    )
                    instance_ids = distances.argmin(dim=1).reshape((-1)).cpu()
                else:
                    instance_ids = torch.zeros(
                        (embedding.shape[1] * embedding.shape[2]),
                        dtype=torch.int64,
                    )
            else:
                instance_ids = instance_ids.cpu()

        else:
            clusterer.fit(embedding_reshaped)
            instance_ids = torch.from_numpy(clusterer.labels_)

        return instance_ids

    def cluster_per_class(self, post_process_conf, out, i, dm, cuml, cudf):
        if post_process_conf.get("type", "dbscan") == "dbscan_gpu":
            data = out["embedding"][i].detach()
            height, width = data.shape[1], data.shape[2]

            instance_ids = torch.zeros(
                (height, width), dtype=torch.int32, device=data.device
            )

            classes = out["semantic"][i].argmax(dim=0)

            for j in range(dm.num_classes):
                if j in dm.mapped_thing_list:
                    class_mask = classes == j

                    embeds = data[:, class_mask]
                    if embeds.shape[1] > 0:
                        clusterer = self.get_clusterer(post_process_conf, cuml, dm)
                        df = cudf.from_dlpack(to_dlpack(embeds.T))

                        clusterer.fit(df)
                        instance_ids_clss = from_dlpack(clusterer.labels_.to_dlpack())
                        instance_ids_clss = remap_instance(instance_ids_clss)

                        instance_ids[class_mask] = instance_ids_clss
            return instance_ids
        else:
            raise Exception("Per class clustering only supported for CUDA DBSCAN")

    def get_clusterer(self, post_process_conf, cuml, dm):
        if post_process_conf.get("type", "dbscan") == "mean_shift":
            clusterer = MeanShift(
                bandwidth=post_process_conf["bandwidth"],
                bin_seeding=post_process_conf["bin_seeding"],
                n_jobs=post_process_conf["n_jobs"],
                max_iter=post_process_conf["max_iter"],
            )
        elif post_process_conf.get("type", "dbscan") == "kmeans":
            clusterer = KMeans(
                n_clusters=dm.label_divisor,
                n_jobs=post_process_conf["n_jobs"],
                n_init=1,
                max_iter=50,
            )
        elif post_process_conf.get("type", "dbscan") == "dbscan_gpu":
            clusterer = cuml.DBSCAN(
                eps=post_process_conf["epsilon"],
                min_samples=post_process_conf["min_samples"],
                calc_core_sample_indices=False,
            )
        else:
            clusterer = DBSCAN(
                eps=post_process_conf["epsilon"],
                min_samples=post_process_conf["min_samples"],
                n_jobs=post_process_conf["n_jobs"],
            )

        return clusterer

    def compute_majority_vote(self, instance_ids, semantic_pred, out, i, dm):
        semantic_pred = semantic_pred.cpu()
        unique_instances = instance_ids.unique()

        instance_onehot = F.one_hot(instance_ids, unique_instances.shape[0])
        semantic_onehot = F.one_hot(semantic_pred, out["semantic"][i].shape[0])
        instance_cls_counts = torch.einsum(
            "hwi,hwc->ic", instance_onehot, semantic_onehot
        )

        instance_class = instance_cls_counts.argmax(dim=1)
        sem_pred = instance_class.gather(0, instance_ids.reshape((-1))).reshape(
            instance_ids.shape
        )

        thing_set = set(dm.mapped_thing_list)
        for i in sem_pred.unique():
            i = i.item()
            if i not in thing_set:
                instance_ids[sem_pred == i] = 0

        panoptic_id = sem_pred * dm.label_divisor + instance_ids

        return panoptic_id

    def plot_prediction(
        self, x, batch, postprocess, out, dm, experiment, batch_idx, global_step
    ):
        tsne_embed = tsne_embedding(out["embedding_small"])

        tsne_embed = F.interpolate(
            tsne_embed,
            size=out["embedding"].shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        grid, panoptic_gt_img, _ = plot_panoptic_prediction(
            x,
            batch["semantic"],
            batch["instance"],
            postprocess["semantic"],
            postprocess["panoptic"],
            None,
            None,
            datamodule=dm,
            extra_imgs=[tsne_embed],
        )

        experiment.add_image(f"predictions-{batch_idx}", grid, global_step)

        for i in range(tsne_embed.shape[0]):
            scatter_plot = plot_embedding_scatter_plot(
                tsne_embed[i], panoptic_gt_img[i]
            )

            experiment.add_image(
                f"embeddings-{batch_idx}-{i}", scatter_plot, global_step
            )
