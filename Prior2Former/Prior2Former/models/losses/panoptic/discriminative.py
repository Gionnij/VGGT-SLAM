import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

# from Stream_Based_AL.utils.import_helper import get_cudf, get_cuml
# import cudf
# import cuml


class DiscriminativeLoss(nn.Module):
    """
    Implements the discriminative loss function as described in https://arxiv.org/pdf/1708.02551.pdf
    """

    def __init__(
        self,
        margin_variance: float = 0.5,
        margin_distance: float = 1.5,
        alpha=1.0,
        beta=1.0,
        gamma=0.001,
        variance_top_k=1.0,
        knn_loss=None,
        per_class_loss=False,
        variance_weights=True,
        distance_variant="margin",
        distance_top_k=1.0,
        clss_center_loss_weight=0.0,
        clss_center_top_k=1.0,
        clss_center_batchwise=False,
        void_instance=False,
    ):
        super(DiscriminativeLoss, self).__init__()

        self.margin_variance = margin_variance
        self.margin_distance = margin_distance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.variance_top_k = variance_top_k
        self.variance_weights = variance_weights
        self.distance_variant = distance_variant
        self.distance_top_k = distance_top_k
        self.clss_center_loss_weight = clss_center_loss_weight
        self.clss_center_top_k = clss_center_top_k
        self.clss_center_batchwise = clss_center_batchwise
        self.void_instance = void_instance

        self.per_class_loss = per_class_loss
        if per_class_loss:
            raise Exception("Per class discriminative loss not yet supported!")

        self.knn_loss = knn_loss

    def forward(self, embeddings, labels, num_classes, **kwargs):
        batch_size = labels["semantic"].shape[0]

        losses = {
            "variance": 0.0,
            "distance": 0.0,
            "regularization": 0.0,
            "total": 0.0,
            "avg_inter_distance": 0.0,
            "avg_intra_distance": 0.0,
            "max_inter_distance": 0.0,
            "min_intra_distance": 0.0,
            "knn": 0.0,
        }

        # cudf = get_cudf()
        # cuml = get_cuml()

        if self.clss_center_loss_weight > 0 and self.clss_center_batchwise:
            class_centers = self.compute_centers_batchwise(
                labels, embeddings, num_classes
            )
        else:
            class_centers = None

        for i in range(batch_size):
            if self.void_instance:
                mask = labels["semantic"][i] != 10000
            else:
                mask = labels["semantic"][i] != 255

            # If we dont check for this, stuff like max wont work
            if mask.sum().cpu().item() > 0:
                embeds = embeddings[i, :, mask]

                instances = labels["instance_remapped"][i][mask]
                instance_onehot = F.one_hot(instances, instances.max() + 1)
                instance_counts = instance_onehot.sum(dim=0)

                centers = torch.einsum(
                    "dn,nc->dc",
                    embeds,
                    instance_onehot.float(),  # embeds.shape=(embeding_dim, num_points), instance_one_hot.shape=(num_points,?num_instances)
                ) / instance_counts.unsqueeze(
                    0
                )  # weighted sum of the embeddings

                variance_loss, distances = self.compute_variance_loss(
                    embeds, instances, centers, instance_counts
                )

                distance_loss, pairwise_distance = self.compute_distance_loss(centers)

                regularization_loss = centers.norm(dim=0).mean()

                if self.clss_center_loss_weight > 0:
                    classes = labels["semantic"][i][mask]

                    center_loss = self.compute_center_loss(
                        classes, embeds, class_centers, num_classes
                    )

                    if "center" not in losses:
                        losses["center"] = 0
                    losses["center"] += center_loss / batch_size

                losses["variance"] += variance_loss / batch_size
                losses["distance"] += distance_loss / batch_size
                losses["regularization"] += regularization_loss / batch_size

                losses["avg_inter_distance"] += distances.mean() / batch_size
                losses["avg_intra_distance"] += pairwise_distance.mean() / batch_size
                losses["max_inter_distance"] += distances.max() / batch_size
                if pairwise_distance.shape[0] > 0:
                    losses["min_intra_distance"] += pairwise_distance.min() / batch_size
                else:
                    losses["min_intra_distance"] += 0

                if self.knn_loss is not None and self.knn_loss["weight"] > 0.0:
                    self.compute_knn_loss(
                        embeds, cudf, cuml, instances, losses, batch_size
                    )
            else:
                losses["regularization"] += (
                    labels["semantic"][i].sum() * 0
                )  # Get backward pass

        if self.knn_loss is not None:
            losses["total"] = (
                self.alpha * losses["variance"]
                + self.beta * losses["distance"]
                + self.gamma * losses["regularization"]
                + self.knn_loss["weight"] * losses["knn"]
            )
        elif self.clss_center_loss_weight > 0:
            losses["total"] = (
                self.alpha * losses["variance"]
                + self.beta * losses["distance"]
                + self.gamma * losses["regularization"]
                + self.clss_center_loss_weight * losses["center"]
            )
        else:
            losses["total"] = (
                self.alpha * losses["variance"]
                + self.beta * losses["distance"]
                + self.gamma * losses["regularization"]
            )

        return losses

    def compute_variance_loss(self, embeds, instances, centers, instance_counts):
        distances = (
            embeds
            - torch.gather(
                centers,
                dim=1,
                index=instances.unsqueeze(0).repeat((centers.shape[0], 1)),
            )
        ).norm(dim=0)
        variance_loss = (distances - self.margin_variance).clip(0)
        variance_loss = variance_loss * variance_loss
        variance_weights = torch.gather(1 / instance_counts, dim=0, index=instances)

        if self.variance_weights:
            variance_loss = variance_loss * variance_weights

        if self.variance_top_k < 1.0:
            variance_loss = variance_loss.topk(
                int(variance_loss.shape[0] * self.variance_top_k)
            ).values.mean()
        else:
            variance_loss = variance_loss.sum() / instance_counts.shape[0]

        return variance_loss, distances

    def compute_distance_loss(self, centers):
        pairwise_distance = F.pdist(centers.transpose(0, 1))
        if self.distance_variant == "margin":
            distance_loss = (2 * self.margin_distance - pairwise_distance).clip(0)
            distance_loss = distance_loss * distance_loss
        elif self.distance_variant == "inverse":
            distance_loss = 1 / (pairwise_distance * pairwise_distance + 0.0001)
        else:
            raise Exception(f"Distance loss variant {self.distance_variant} not known")

        if self.distance_top_k < 1:
            distance_loss = distance_loss.topk(
                int(distance_loss.shape[0] * self.distance_top_k)
            ).values

        if distance_loss.shape[0] > 0:
            distance_loss = distance_loss.mean()
        else:
            distance_loss = 0

        return distance_loss, pairwise_distance

    def compute_centers_batchwise(self, labels, embeddings, num_classes):
        batch_size = labels["semantic"].shape[0]

        nc = num_classes + (1 if self.void_instance else 0)

        total_class_counts = torch.zeros(
            (nc,), dtype=torch.int64, device=embeddings.device
        )
        class_embeds = torch.zeros((nc, embeddings.shape[1]), device=embeddings.device)

        for i in range(batch_size):
            if self.void_instance:
                mask = labels["semantic"][i] != 10000
            else:
                mask = labels["semantic"][i] != 255
            embeds = embeddings[i, :, mask]

            # If we dont check for this, stuff like max wont work
            if mask.sum().cpu().item() > 0:
                classes = labels["semantic"][i][mask]

                if self.void_instance:
                    classes = classes.clone()
                    classes[classes == 255] = num_classes

                clss_onehot = F.one_hot(classes, num_classes=nc)
                clss_counts = clss_onehot.sum(dim=0)

                class_centers = torch.einsum("dn,nc->cd", embeds, clss_onehot.float())

                total_class_counts = total_class_counts + clss_counts
                class_embeds = class_embeds + class_centers

        return class_embeds / (total_class_counts.unsqueeze(1) + 1)

    def compute_center_loss(self, classes, embeds, class_centers, num_classes):
        if self.void_instance:
            classes = classes.clone()
            classes[classes == 255] = num_classes

        if class_centers is None:
            clss_onehot = F.one_hot(
                classes, num_classes=num_classes + 1 if self.void_instance else 0
            )
            clss_counts = clss_onehot.sum(dim=0)

            class_centers = torch.einsum("dn,nc->cd", embeds, clss_onehot.float()) / (
                clss_counts.unsqueeze(1) + 1
            )
            # We add 1 for numeric stability, otherwise classes with class counts 0 will a NaN loss

        center_dists = (
            class_centers.gather(
                dim=0,
                index=(
                    classes.reshape((-1, 1)).expand(
                        (classes.shape[0], class_centers.shape[1])
                    )
                ),
            )
            - embeds.T
        ).norm(dim=1)
        center_loss = (center_dists - 0.0).clip(0)
        center_loss = center_loss * center_loss

        if self.clss_center_top_k < 1.0:
            center_loss = center_loss.topk(
                int(center_loss.shape[0] * self.clss_center_top_k)
            ).values

        center_loss = center_loss.mean()

        return center_loss

    def compute_knn_loss(self, embeds, cudf, cuml, instances, losses, batch_size):
        embeds_T = embeds.T

        start = time.time()

        df = cudf.from_dlpack(to_dlpack(embeds_T))

        nn = cuml.NearestNeighbors(**self.knn_loss["knn_args"])
        nn.fit(df)
        indices = nn.kneighbors(df, return_distance=False)

        indices = from_dlpack(indices.to_dlpack())

        end = time.time()

        n_neighbors = indices.shape[1] - 1

        ind1 = (
            indices[:, 0].reshape((-1, 1)).expand((indices.shape[0], n_neighbors))
        ).reshape((-1))
        ind2 = indices[:, 1:].reshape((-1))

        instance_1 = instances.gather(dim=0, index=ind1)
        instance_2 = instances.gather(dim=0, index=ind2)

        mask_diff = instance_1 != instance_2

        ind1 = ind1[mask_diff]
        ind2 = ind2[mask_diff]

        n_pairs = ind1.shape[0]

        if n_pairs > 0:
            embeds_1 = embeds_T.gather(
                dim=0,
                index=ind1.reshape((-1, 1)).expand((n_pairs, embeds_T.shape[1])),
            )
            embeds_2 = embeds_T.gather(
                dim=0,
                index=ind2.reshape((-1, 1)).expand((n_pairs, embeds_T.shape[1])),
            )

            dists = F.pairwise_distance(embeds_1, embeds_2)
            knn_loss = (self.knn_loss["margin"] - dists).clip(0)
            knn_loss = (knn_loss * knn_loss).mean()
            losses["knn"] += knn_loss / batch_size


class DiscriminativeLoss_CosineSim(nn.Module):
    """
    Implements the discriminative loss function as described in https://arxiv.org/pdf/1708.02551.pdf
    """

    def __init__(
        self,
        margin_variance: float = 0.5,
        margin_distance: float = 1.5,
        alpha=1.0,
        beta=1.0,
        gamma=0.001,
        variance_top_k=1.0,
        knn_loss=None,
        per_class_loss=False,
        variance_weights=True,
        distance_variant="margin",
        distance_top_k=1.0,
        clss_center_loss_weight=0.0,
        clss_center_top_k=1.0,
        clss_center_batchwise=False,
        void_instance=False,
    ):
        super(DiscriminativeLoss, self).__init__()

        self.margin_variance = margin_variance
        self.margin_distance = margin_distance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.variance_top_k = variance_top_k
        self.variance_weights = variance_weights
        self.distance_variant = distance_variant
        self.distance_top_k = distance_top_k
        self.clss_center_loss_weight = clss_center_loss_weight
        self.clss_center_top_k = clss_center_top_k
        self.clss_center_batchwise = clss_center_batchwise
        self.void_instance = void_instance

        self.per_class_loss = per_class_loss
        if per_class_loss:
            raise Exception("Per class discriminative loss not yet supported!")

        self.knn_loss = knn_loss

    def forward(self, embeddings, labels, num_classes, **kwargs):
        batch_size = labels["semantic"].shape[0]

        losses = {
            "variance": 0.0,
            "distance": 0.0,
            "regularization": 0.0,
            "total": 0.0,
            "avg_inter_distance": 0.0,
            "avg_intra_distance": 0.0,
            "max_inter_distance": 0.0,
            "min_intra_distance": 0.0,
            "knn": 0.0,
        }

        # cudf = get_cudf()
        # cuml = get_cuml()

        if self.clss_center_loss_weight > 0 and self.clss_center_batchwise:
            class_centers = self.compute_centers_batchwise(
                labels, embeddings, num_classes
            )
        else:
            class_centers = None

        for i in range(batch_size):
            if self.void_instance:
                mask = labels["semantic"][i] != 10000
            else:
                mask = labels["semantic"][i] != 255

            # If we dont check for this, stuff like max wont work
            if mask.sum().cpu().item() > 0:
                embeds = embeddings[i, :, mask]

                instances = labels["instance_remapped"][i][mask]
                instance_onehot = F.one_hot(instances, instances.max() + 1)
                instance_counts = instance_onehot.sum(dim=0)

                centers = torch.einsum(
                    "dn,nc->dc",
                    embeds,
                    instance_onehot.float(),  # embeds.shape=(embeding_dim, num_points), instance_one_hot.shape=(num_points,?num_instances)
                ) / instance_counts.unsqueeze(
                    0
                )  # weighted sum of the embeddings

                variance_loss, distances = self.compute_variance_loss(
                    embeds, instances, centers, instance_counts
                )

                distance_loss, pairwise_distance = self.compute_distance_loss(centers)

                regularization_loss = centers.norm(dim=0).mean()

                if self.clss_center_loss_weight > 0:
                    classes = labels["semantic"][i][mask]

                    center_loss = self.compute_center_loss(
                        classes, embeds, class_centers, num_classes
                    )

                    if "center" not in losses:
                        losses["center"] = 0
                    losses["center"] += center_loss / batch_size

                losses["variance"] += variance_loss / batch_size
                losses["distance"] += distance_loss / batch_size
                losses["regularization"] += regularization_loss / batch_size

                losses["avg_inter_distance"] += distances.mean() / batch_size
                losses["avg_intra_distance"] += pairwise_distance.mean() / batch_size
                losses["max_inter_distance"] += distances.max() / batch_size
                if pairwise_distance.shape[0] > 0:
                    losses["min_intra_distance"] += pairwise_distance.min() / batch_size
                else:
                    losses["min_intra_distance"] += 0

                if self.knn_loss is not None and self.knn_loss["weight"] > 0.0:
                    self.compute_knn_loss(
                        embeds, cudf, cuml, instances, losses, batch_size
                    )
            else:
                losses["regularization"] += (
                    labels["semantic"][i].sum() * 0
                )  # Get backward pass

        if self.knn_loss is not None:
            losses["total"] = (
                self.alpha * losses["variance"]
                + self.beta * losses["distance"]
                + self.gamma * losses["regularization"]
                + self.knn_loss["weight"] * losses["knn"]
            )
        elif self.clss_center_loss_weight > 0:
            losses["total"] = (
                self.alpha * losses["variance"]
                + self.beta * losses["distance"]
                + self.gamma * losses["regularization"]
                + self.clss_center_loss_weight * losses["center"]
            )
        else:
            losses["total"] = (
                self.alpha * losses["variance"]
                + self.beta * losses["distance"]
                + self.gamma * losses["regularization"]
            )

        return losses

    def compute_variance_loss(self, embeds, instances, centers, instance_counts):
        distances = (
            embeds
            - torch.gather(
                centers,
                dim=1,
                index=instances.unsqueeze(0).repeat((centers.shape[0], 1)),
            )
        ).norm(dim=0)
        variance_loss = (distances - self.margin_variance).clip(0)
        variance_loss = variance_loss * variance_loss
        variance_weights = torch.gather(1 / instance_counts, dim=0, index=instances)

        if self.variance_weights:
            variance_loss = variance_loss * variance_weights

        if self.variance_top_k < 1.0:
            variance_loss = variance_loss.topk(
                int(variance_loss.shape[0] * self.variance_top_k)
            ).values.mean()
        else:
            variance_loss = variance_loss.sum() / instance_counts.shape[0]

        return variance_loss, distances

    def compute_distance_loss(self, centers):
        pairwise_distance = F.pdist(centers.transpose(0, 1))
        if self.distance_variant == "margin":
            distance_loss = (2 * self.margin_distance - pairwise_distance).clip(0)
            distance_loss = distance_loss * distance_loss
        elif self.distance_variant == "inverse":
            distance_loss = 1 / (pairwise_distance * pairwise_distance + 0.0001)
        else:
            raise Exception(f"Distance loss variant {self.distance_variant} not known")

        if self.distance_top_k < 1:
            distance_loss = distance_loss.topk(
                int(distance_loss.shape[0] * self.distance_top_k)
            ).values

        if distance_loss.shape[0] > 0:
            distance_loss = distance_loss.mean()
        else:
            distance_loss = 0

        return distance_loss, pairwise_distance

    def compute_centers_batchwise(self, labels, embeddings, num_classes):
        batch_size = labels["semantic"].shape[0]

        nc = num_classes + (1 if self.void_instance else 0)

        total_class_counts = torch.zeros(
            (nc,), dtype=torch.int64, device=embeddings.device
        )
        class_embeds = torch.zeros((nc, embeddings.shape[1]), device=embeddings.device)

        for i in range(batch_size):
            if self.void_instance:
                mask = labels["semantic"][i] != 10000
            else:
                mask = labels["semantic"][i] != 255
            embeds = embeddings[i, :, mask]

            # If we dont check for this, stuff like max wont work
            if mask.sum().cpu().item() > 0:
                classes = labels["semantic"][i][mask]

                if self.void_instance:
                    classes = classes.clone()
                    classes[classes == 255] = num_classes

                clss_onehot = F.one_hot(classes, num_classes=nc)
                clss_counts = clss_onehot.sum(dim=0)

                class_centers = torch.einsum("dn,nc->cd", embeds, clss_onehot.float())

                total_class_counts = total_class_counts + clss_counts
                class_embeds = class_embeds + class_centers

        return class_embeds / (total_class_counts.unsqueeze(1) + 1)

    def compute_center_loss(self, classes, embeds, class_centers, num_classes):
        if self.void_instance:
            classes = classes.clone()
            classes[classes == 255] = num_classes

        if class_centers is None:
            clss_onehot = F.one_hot(
                classes, num_classes=num_classes + 1 if self.void_instance else 0
            )
            clss_counts = clss_onehot.sum(dim=0)

            class_centers = torch.einsum("dn,nc->cd", embeds, clss_onehot.float()) / (
                clss_counts.unsqueeze(1) + 1
            )
            # We add 1 for numeric stability, otherwise classes with class counts 0 will a NaN loss

        center_dists = (
            class_centers.gather(
                dim=0,
                index=(
                    classes.reshape((-1, 1)).expand(
                        (classes.shape[0], class_centers.shape[1])
                    )
                ),
            )
            - embeds.T
        ).norm(dim=1)
        center_loss = (center_dists - 0.0).clip(0)
        center_loss = center_loss * center_loss

        if self.clss_center_top_k < 1.0:
            center_loss = center_loss.topk(
                int(center_loss.shape[0] * self.clss_center_top_k)
            ).values

        center_loss = center_loss.mean()

        return center_loss

    def compute_knn_loss(self, embeds, cudf, cuml, instances, losses, batch_size):
        embeds_T = embeds.T

        start = time.time()

        df = cudf.from_dlpack(to_dlpack(embeds_T))

        nn = cuml.NearestNeighbors(**self.knn_loss["knn_args"])
        nn.fit(df)
        indices = nn.kneighbors(df, return_distance=False)

        indices = from_dlpack(indices.to_dlpack())

        end = time.time()

        n_neighbors = indices.shape[1] - 1

        ind1 = (
            indices[:, 0].reshape((-1, 1)).expand((indices.shape[0], n_neighbors))
        ).reshape((-1))
        ind2 = indices[:, 1:].reshape((-1))

        instance_1 = instances.gather(dim=0, index=ind1)
        instance_2 = instances.gather(dim=0, index=ind2)

        mask_diff = instance_1 != instance_2

        ind1 = ind1[mask_diff]
        ind2 = ind2[mask_diff]

        n_pairs = ind1.shape[0]

        if n_pairs > 0:
            embeds_1 = embeds_T.gather(
                dim=0,
                index=ind1.reshape((-1, 1)).expand((n_pairs, embeds_T.shape[1])),
            )
            embeds_2 = embeds_T.gather(
                dim=0,
                index=ind2.reshape((-1, 1)).expand((n_pairs, embeds_T.shape[1])),
            )

            dists = F.pairwise_distance(embeds_1, embeds_2)
            knn_loss = (self.knn_loss["margin"] - dists).clip(0)
            knn_loss = (knn_loss * knn_loss).mean()
            losses["knn"] += knn_loss / batch_size
