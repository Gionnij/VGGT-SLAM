import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEmbeddingLoss(nn.Module):

    def __init__(
        self,
        void_weight: float = 0.5,
        known_weight: float = 0.5,
        ignore_label: int = 255,
        hinge: bool = True,
    ):
        super(ContrastiveEmbeddingLoss, self).__init__()
        self.void_weight = void_weight
        self.known_weight = torch.tensor(known_weight)
        self.ignore_label = torch.tensor(ignore_label)
        self.hinge = hinge

    def forward(self, embeddings, labels, num_classes, **kwargs):
        device = embeddings.device
        batch_size = labels["semantic"].shape[0]
        semantic = (
            F.interpolate(
                labels["semantic"].unsqueeze(1).to(float),
                size=embeddings.shape[-2:],
                mode="nearest",
            )
            .squeeze(1)
            .to(int)
        )
        loss = 0
        for i in range(batch_size):

            mask = semantic[i] != self.ignore_label

            embeds_known = embeddings[i, :, mask] if mask.any() else torch.tensor(1.0)
            embeds_void = (
                embeddings[i, :, ~mask] if (~mask).any() else torch.tensor(0.0)
            )

            # Hinge loss for embeds_known
            if self.hinge:
                loss += torch.clamp(1 - embeds_known.norm(2, dim=0), min=0).mean().to(
                    device
                ) * self.known_weight.to(device)
            else:
                loss += (
                    1 - embeds_known.norm(2, dim=0).mean().to(device)
                ) ** 2 * self.known_weight.to(device)

            loss += embeds_void.norm(2, dim=0).mean() * self.void_weight

        return loss / 2 / batch_size
