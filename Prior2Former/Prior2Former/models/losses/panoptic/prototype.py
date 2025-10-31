import torch
import torch.nn as nn
from torch.nn import functional as F

EPSILON = 0.000001


class PrototypeInstanceLoss(nn.Module):
    def __init__(
        self,
        spatial_clustering: bool = False,
        random_pos_offset=None,
        bias_weight=1.0,
        mixture_dist="normal",
        hard_pixel_fraction=1.0,
        predict_sigmas=True,
    ):
        super(PrototypeInstanceLoss, self).__init__()

        self.spatial_clustering = spatial_clustering

        self.num_sigmas = 2 if spatial_clustering else 1
        self.random_pos_offset = random_pos_offset
        self.bias_weight = bias_weight
        self.mixture_dist = mixture_dist
        self.hard_pixel_fraction = hard_pixel_fraction
        self.predict_sigmas = predict_sigmas

    def forward(self, results, targets, no_class_cfg=None):
        thing_prototypes = results["thing_prototypes"]
        instance_positions = targets["center_positions"]
        num_instances = targets["num_instances"]

        height, width = thing_prototypes.shape[2], thing_prototypes.shape[3]

        batch_size = thing_prototypes.shape[0]

        total_loss = 0

        for i in range(batch_size):
            n_instances = num_instances[i].cpu().item()
            instance_position = instance_positions[i][:n_instances]

            if self.random_pos_offset is not None:
                instance_position = self.apply_random_pos_offset(
                    instance_position, height, width
                )

            instance_prototypes = (
                thing_prototypes[i]
                .contiguous()
                .reshape((-1, height * width))
                .gather(
                    dim=1,
                    index=(instance_position[:, 0] * width + instance_position[:, 1])
                    .reshape((1, n_instances))
                    .expand((thing_prototypes[i].shape[0], n_instances)),
                )
            )
            stuff_prototypes = results["stuff_prototypes"][i]

            all_prototypes = torch.cat(
                [instance_prototypes, stuff_prototypes], dim=1
            ).transpose(0, 1)

            embeddings = results["embedding"][i]
            embedding_dim = embeddings.shape[0]

            dists = torch.cdist(
                all_prototypes[None, :, :embedding_dim],
                embeddings.reshape((embedding_dim, -1)).transpose(0, 1).unsqueeze(0),
            ).squeeze(0)
            if self.predict_sigmas:
                sigmas = all_prototypes[:, embedding_dim : embedding_dim + 1]
            else:
                sigmas = torch.ones_like(
                    all_prototypes[:, embedding_dim - 1 : embedding_dim]
                )
            yhat = self.calculate_yhat(dists, sigmas)

            if self.spatial_clustering:
                yhat2 = self.calculate_pixel_distances(
                    all_prototypes,
                    instance_position,
                    embedding_dim,
                    width,
                    height,
                    batch=i,
                )
                yhat = yhat + yhat2

            if no_class_cfg is not None:
                yhat = torch.cat(
                    (
                        yhat,
                        no_class_cfg["no_class_score"]
                        .reshape((1, 1))
                        .repeat((1, yhat.shape[1])),
                    ),
                    dim=0,
                )

            gt_assocs = targets["center_assoc"][i, 0].clone()

            stuff_mask = (gt_assocs == -1) & (targets["stuff_id"][i, 0] != 255)
            gt_assocs[stuff_mask] = (
                targets["stuff_id"][i, 0, stuff_mask].reshape((-1)) + n_instances
            )

            if no_class_cfg is not None and no_class_cfg.get("use_void", False):
                gt_assocs[targets["semantic"][i] == 255] = yhat.shape[0] - 1

            gt_assocs[gt_assocs == -1] = 10000

            gt_assocs = gt_assocs.reshape((-1))
            mask = gt_assocs != 10000

            loss = self.compute_loss(yhat[:, mask], gt_assocs[mask])

            if self.hard_pixel_fraction < 1:
                num_k = int(loss.shape[0] * self.hard_pixel_fraction)
                loss, _ = torch.topk(loss, num_k)
            total_loss += loss.mean() / batch_size
            if total_loss.detach().cpu().isnan().item():
                print("instance loss is NaN")
                return torch.tensor(0)
        return total_loss

    def calculate_pixel_distances(
        self, all_prototypes, instance_positions, embedding_dim, width, height, batch
    ):
        sigmas = all_prototypes[:, -self.num_sigmas + 1 :]
        height_pos = (
            torch.arange(0, height, 1, device=instance_positions.device)
            .reshape((height, 1, 1))
            .expand(height, width, 1)
        ) / (height - 1)
        width_pos = (
            torch.arange(0, width, 1, device=instance_positions.device)
            .reshape((1, width, 1))
            .expand(height, width, 1)
        ) / (width - 1)
        pixel_positions = torch.cat((height_pos, width_pos), dim=2).reshape((-1, 2))

        instance_positions = instance_positions / torch.tensor(
            [height - 1, width - 1], device=instance_positions.device
        )

        dists = torch.cdist(
            instance_positions.unsqueeze(0).float(),
            pixel_positions.unsqueeze(0).float(),
        ).squeeze(0)

        yhat = self.calculate_yhat(
            F.pad(dists, [0, 0, 0, sigmas.shape[0] - dists.shape[0]]), sigmas
        )

        return yhat

    def calculate_yhat(self, dists, sigmas):
        embedding_dim = dists.shape[0] - self.num_sigmas
        if self.mixture_dist == "normal":
            return -dists * dists / (sigmas * 2 + EPSILON) - (
                embedding_dim / 2 * torch.log(sigmas) if self.bias_weight == 1 else 0.0
            )
        elif self.mixture_dist == "cauchy":
            return -torch.log((1 + dists * dists / (sigmas + EPSILON))) - (
                embedding_dim * torch.log(sigmas) if self.bias_weight == 1 else 0.0
            )
        elif self.mixture_dist == "dirichlet":
            if self.bias_weight == 1:
                raise Exception(
                    "Using the dirichlet mixture distribution implies no bias weight"
                )
            return (sigmas * 2) / (dists * dists + EPSILON)
        raise Exception(f"Mixture distribution named {self.mixture_dist} not known")

    def compute_loss(self, yhat, gt_assocs):
        if self.mixture_dist == "dirichlet":
            evidence = yhat
            alpha = evidence + 1
            S = alpha.sum(dim=0, keepdim=True)

            yc = F.one_hot(gt_assocs, alpha.shape[0]).permute(1, 0)

            return (yc * (torch.log(S) - torch.log(alpha))).sum(dim=0, keepdim=False)
        else:
            return F.cross_entropy(
                yhat.unsqueeze(0),
                gt_assocs.unsqueeze(0),
                ignore_index=10000,
                reduction="none",
            ).squeeze(0)

    def apply_random_pos_offset(self, instance_position, height, width):
        offsets = torch.randint(
            -self.random_pos_offset,
            self.random_pos_offset,
            instance_position.shape,
            device=instance_position.device,
        )
        new_instance_position = instance_position + offsets
        new_instance_position[:, 0] = torch.clip(
            new_instance_position[:, 0], 0, height - 1
        )
        new_instance_position[:, 1] = torch.clip(
            new_instance_position[:, 1], 0, width - 1
        )

        return new_instance_position
