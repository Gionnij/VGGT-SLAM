import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeCosineEmbeddingLoss(nn.Module):

    def __init__(
        self,
        margin_cluster: float = 0,
        margin_cluster_mean: float = 0,
        within_cluster_weigths: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
        ignore_label: int = 255,
        reg: float = 0.0,
        reg_type: str = "center",
    ):
        super(DiscriminativeCosineEmbeddingLoss, self).__init__()
        self.margin_cluster = margin_cluster
        self.within_cluster_weigths = within_cluster_weigths
        self.margin_cluster_mean = margin_cluster_mean
        self.alpha = alpha
        self.beta = beta
        self.ignore_label = ignore_label
        self.reg = reg
        self.reg_type = reg_type

    def compute_within_cluster_similarity(
        self, embeds, instances, centers, instance_counts
    ):
        similarities = F.cosine_similarity(
            embeds,
            torch.gather(
                centers,
                dim=1,
                index=instances.unsqueeze(0).repeat((centers.shape[0], 1)),
            ),
            dim=0,
        )
        sim_loss = (1 - similarities - self.margin_cluster).clip(0)
        sim_weights = torch.gather(1 / instance_counts, dim=0, index=instances)
        if self.within_cluster_weigths:
            sim_loss = sim_loss * sim_weights
        sim_loss = sim_loss.sum() / instance_counts.shape[0]
        # if sim_loss.isnan().item():
        #     sim_loss = torch.tensor(0)
        return sim_loss, similarities

    def compute_cluster_center_loss(self, centers):
        # centers = centers[
        #     :, ~centers[0, :].isnan()
        # ]  # if instance count is 0 the center are nan in that embedding
        centers_norm = F.normalize(centers, p=2, dim=0)
        pairwise_sim = torch.mm(centers_norm.t(), centers_norm)
        sim_loss = (1 + pairwise_sim - self.margin_cluster_mean).clip(0)
        sim_loss = sim_loss[
            ~torch.eye(sim_loss.shape[0]).to(sim_loss).to(bool)
        ]  # take out diagonal elements for backpropagation
        sim_loss = sim_loss.mean()
        # if sim_loss.isnan().item():
        #     sim_loss = torch.tensor(0)
        return sim_loss, pairwise_sim

    def regularization(self, embeddings, centers, reg_type):
        if reg_type == "center":
            # centers = centers[
            #     :, ~centers[0, :].isnan()
            # ]  # if instance count is 0 the center are nan in that embedding
            return ((1 - centers.norm(2, dim=0)) ** 2).mean()
        elif reg_type == "embedding":
            return ((1 - embeddings.norm(2, dim=0)) ** 2).mean()
        else:
            raise NotImplementedError(
                f"regularization for the embedding of type {reg_type} is unkonw"
            )

    def forward(self, embeddings, labels, num_classes, **kwargs):
        batch_size = labels["semantic"].shape[0]
        h, w = embeddings.shape[-2:]
        semantic = (
            F.interpolate(
                labels["semantic"].unsqueeze(1).to(float),
                size=embeddings.shape[-2:],
                mode="nearest",
            )
            .squeeze(1)
            .to(int)
        )
        instances_batch = (
            F.interpolate(
                labels["instance_remapped"].unsqueeze(1).to(float),
                size=embeddings.shape[-2:],
                mode="nearest",
            )
            .squeeze(1)
            .to(int)
        )
        for i in range(batch_size):

            mask = semantic[i] != self.ignore_label

            # If we dont check for this, stuff like max wont work
            if mask.sum().cpu().item() > 0:
                embeds = embeddings[i, :, mask]

                instances = instances_batch[i][mask]
                instance_onehot = F.one_hot(instances, instances.max() + 1)
                instance_counts = instance_onehot.sum(dim=0)

                centers = torch.einsum(
                    "dn,nc->dc",
                    embeds,
                    instance_onehot.float(),  # embeds.shape=(embeding_dim, num_points), instance_one_hot.shape=(num_points,?num_instances)
                ) / instance_counts.unsqueeze(
                    0
                )  # weighted sum of the embeddings

                cosine_loss, similarities = self.compute_within_cluster_similarity(
                    embeds, instances, centers, instance_counts
                )

                pairwise_loss, pairwise_sim = self.compute_cluster_center_loss(centers)

                if self.reg > 0:
                    regularization = self.regularization(embeds, centers, self.reg_type)
                else:
                    regularization = 0

                if (
                    (
                        self.alpha * cosine_loss
                        + self.beta * pairwise_loss
                        + self.reg * regularization
                    )
                    .isnan()
                    .item()
                ):
                    print("Discriminative loss is nan!")
                    return {
                        "total": torch.tensor(0),
                        "similarity": similarities,
                        "pairwise_similarity": pairwise_sim,
                    }
                return {
                    "total": self.alpha * cosine_loss
                    + self.beta * pairwise_loss
                    + self.reg * regularization,
                    "similarity": similarities,
                    "pairwise_similarity": pairwise_sim,
                }
