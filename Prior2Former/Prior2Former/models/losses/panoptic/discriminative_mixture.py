import torch
import torch.nn as nn
from torch.nn import functional as F


class DiscriminativeMixtureLoss(nn.Module):
    def __init__(
        self,
        sigma=1.0,
        mixture_dist="normal",
        gamma=0.001,
        hard_pixel_fraction=1.0,
        use_semantic_weights=False,
        sigma_other=None,
        void_instance=False,
        class_loss=None,
        per_class=False,
    ):
        super(DiscriminativeMixtureLoss, self).__init__()
        self.sigma = sigma
        self.mixture_dist = mixture_dist
        self.gamma = gamma
        self.hard_pixel_fraction = hard_pixel_fraction
        self.use_semantic_weights = use_semantic_weights
        self.sigma_other = sigma_other
        self.void_instance = void_instance
        self.class_loss = class_loss
        self.per_class = per_class

    def forward(self, embeddings, labels, num_classes, dm, **kwargs):
        batch_size = labels["semantic"].shape[0]

        losses = {
            "log_prob": 0.0,
            "regularization": 0.0,
            "total": 0.0,
        }

        if (
            self.class_loss is not None
            and "batchwise" in self.class_loss
            and self.class_loss["batchwise"]
        ):
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

                ce_loss = self.compute_ce_loss(labels, embeds, i, mask, dm)

                regularization_loss = embeds.norm(dim=0)
                if self.hard_pixel_fraction < 1.0:
                    num_k = int(ce_loss.shape[0] * self.hard_pixel_fraction)
                    ce_loss, indices = torch.topk(ce_loss, num_k)
                    regularization_loss = regularization_loss[indices]

                losses["log_prob"] = ce_loss.mean() / batch_size
                losses["regularization"] += regularization_loss.mean() / batch_size

                if self.class_loss is not None:
                    classes = labels["semantic"][i][mask]
                    class_center_loss = self.compute_class_center_loss(
                        classes, class_centers, num_classes, embeds
                    )

                    if "class_center" not in losses:
                        losses["class_center"] = 0
                    losses["class_center"] += class_center_loss / batch_size

            else:
                losses["regularization"] += (
                    labels["semantic"][i].sum() * 0
                )  # Get backward pass

        losses["total"] = losses["log_prob"] + self.gamma * losses["regularization"]
        if self.class_loss is not None:
            losses["total"] += losses["class_center"] * self.class_loss["weight"]

        return losses

    def log_probs(self, distances, sigma):
        if self.mixture_dist == "normal":
            return -distances / sigma
        elif self.mixture_dist == "cauchy":
            return -torch.log((1 + distances * distances / sigma))
        else:
            raise Exception(f"Mixture distribution {self.mixture_dist} unknown")

    def compute_ce_loss(self, labels, embeds, i, mask, dm):
        instances = labels["instance_remapped"][i][mask]
        instance_onehot = F.one_hot(instances, instances.max() + 1)
        instance_counts = instance_onehot.sum(dim=0)

        centers = torch.einsum(
            "dn,nc->dc", embeds, instance_onehot.float()
        ) / instance_counts.unsqueeze(0)

        if self.per_class:
            total_loss = torch.tensor(0.0, device=centers.device)
            for j in dm.mapped_thing_list:
                mask_class = labels["semantic"][i, mask] == j

                instance_class = instances[mask_class]

                if instance_class.shape[0] > 0:
                    instance_class_unique = instance_class.unique()
                    centers_class = centers.gather(
                        dim=1,
                        index=instance_class_unique.reshape((1, -1)).expand(
                            (centers.shape[0], instance_class_unique.shape[0])
                        ),
                    )

                    distances = (
                        torch.cdist(
                            centers_class.transpose(0, 1).unsqueeze(0).contiguous(),
                            embeds[:, mask_class]
                            .transpose(0, 1)
                            .unsqueeze(0)
                            .contiguous(),
                        )
                        .squeeze(0)
                        .contiguous()
                    )

                    log_probs = self.log_probs(distances, self.sigma)

                    instance_remapped = (
                        (
                            instance_class_unique.unsqueeze(1)
                            == instance_class.unsqueeze(0)
                        )
                        * 1
                    ).argmax(dim=0)
                    ce_loss = F.cross_entropy(
                        log_probs.unsqueeze(0), instance_remapped.unsqueeze(0)
                    )

                    total_loss += ce_loss / len(dm.mapped_thing_list)
            return total_loss
        else:
            distances = torch.cdist(
                centers.transpose(0, 1).unsqueeze(0),
                embeds.transpose(0, 1).unsqueeze(0),
            ).squeeze(0)

            if self.sigma_other is not None:
                instance_clss = labels["instance_clss"][i][: instance_onehot.shape[1]]
                same_clss = labels["semantic"][i][mask].unsqueeze(
                    0
                ) == instance_clss.unsqueeze(1)
                log_probs = torch.zeros_like(distances)
                log_probs[same_clss] = self.log_probs(distances[same_clss], self.sigma)
                log_probs[~same_clss] = self.log_probs(
                    distances[~same_clss], self.sigma_other
                )
            else:
                log_probs = self.log_probs(distances, self.sigma)

            ce_loss = F.cross_entropy(
                log_probs.unsqueeze(0), instances.unsqueeze(0), reduction="none"
            ).squeeze(0)

            if self.use_semantic_weights and "semantic_weights" in labels:
                semantic_weights = labels["semantic_weights"][i][mask]
                ce_loss = ce_loss * semantic_weights

            return ce_loss

    def compute_class_center_loss(self, classes, class_centers, num_classes, embeds):
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

        center_dists = torch.cdist(
            class_centers.unsqueeze(0), embeds.T.unsqueeze(0)
        ).squeeze(0)
        center_loss = -torch.log(
            (1 + center_dists * center_dists / self.class_loss["sigma"])
        )
        center_loss = F.cross_entropy(center_loss.unsqueeze(0), classes.unsqueeze(0))

        return center_loss

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
