import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepLabV3Base.deeplabv3.panopticdl.decoder import SinglePanopticDeepLabHead
from models.DeepLabV3Base.deeplabv3.util import apply_nonlinearity



class DPNHead(SinglePanopticDeepLabHead):
    def __init__(self, nonlinearity: str, num_classes: int, *args, **kwargs):
        super().__init__(
            num_classes=[num_classes], class_key=["result"], *args, **kwargs
        )

        self.nonlinearity = nonlinearity

    def forward(self, x):
        logits = super().forward(x)["result"]
        x = apply_nonlinearity(self.nonlinearity, logits)
        return {"logits": logits, "semantic": x}

    def get_uncertainty(self, out, i):
        evidence = out["semantic"][i]
        alpha = evidence + 1
        alpha_sum = alpha.sum(dim=0)
        uncertainty = alpha.shape[0] / alpha_sum
        return uncertainty

    def training_step(self, x, out):
        pass


class DUQHead(SinglePanopticDeepLabHead):
    def __init__(
        self,
        feature_dim: int,
        centroid_size: int,
        num_classes: int,
        length_scale=0.1,
        gamma=0.999,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_classes=[feature_dim], class_key=["result"], *args, **kwargs
        )

        self.n_classes = num_classes

        self.W = nn.Parameter(torch.zeros(centroid_size, num_classes, feature_dim))
        self.register_buffer("N", torch.zeros(num_classes) + 13)
        m = torch.normal(
            torch.zeros(centroid_size, num_classes),
            # This makes sure the centroids have norm approximately 1 after initialization
            1 / np.sqrt(centroid_size),
        )
        self.register_buffer("m", m)
        self.m = self.m * self.N

        self.sigma = length_scale
        self.gamma = gamma

    def forward(self, x):
        features = super().forward(x)["result"]

        y_pred, z = self.rbf(features)

        return {"logits": y_pred, "semantic": y_pred, "features": features, "z": z}

    def rbf(self, z):
        z = torch.einsum("bfwh,znf->bznwh", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0).unsqueeze(3).unsqueeze(4)

        diff = (diff**2).mean(1).div(-2 * self.sigma**2).exp()

        return diff, z

    def get_uncertainty(self, out, i):
        return 1 - out["semantic"][i].max(dim=0).values

    def training_step(self, x, out):
        self.update_embeddings(x, out)

    def update_embeddings(self, batch, out):
        with torch.no_grad():
            y = batch["semantic"]
            z = out["semantic_additional"]["z"]

            yc = (
                F.interpolate(
                    y.unsqueeze(1).float(),
                    size=z.shape[3:],
                    mode="nearest",
                )
                .squeeze()
                .long()
            )
            yc[yc == 255] = self.n_classes
            yc = F.one_hot(yc, self.n_classes + 1)[:, :, :, : self.n_classes]
            yc = yc.permute(0, 3, 1, 2)

            self.N = self.gamma * self.N + (1 - self.gamma) * yc.sum((0, 2, 3))

            embedding_sum = torch.einsum("bznhw,bnhw->zn", z.detach(), yc.float())

            self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum


class SNGPHead(SinglePanopticDeepLabHead):
    def __init__(
        self,
        feature_dim,
        rf_dim=64,
        num_classes=21,
        gamma=0.99,
        mc_samples=10,
        uncertainty_mode="probs",
        *args,
        **kwargs,
    ):
        super().__init__(
            num_classes=[feature_dim], class_key=["result"], *args, **kwargs
        )

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.rf_dim = rf_dim

        D = self.rf_dim

        W = torch.normal(torch.zeros(feature_dim, D), 1 / np.sqrt(feature_dim))
        self.register_buffer("W", W)

        self.register_buffer("b", torch.rand(D) * np.pi)
        self.beta = nn.Parameter(torch.normal(torch.zeros(self.rf_dim, num_classes), 1))

        self.register_buffer(
            "precision",
            torch.eye(rf_dim).reshape((1, rf_dim, rf_dim)).repeat((num_classes, 1, 1)),
        )

        self.gamma = gamma

        self.mc_samples = mc_samples
        self.covariance = None
        self.uncertainty_mode = uncertainty_mode

    def forward(self, x):
        features = super().forward(x)["result"]

        z = torch.einsum("bfhw,fz->bzhw", features, self.W)

        phi = torch.cos(-z + self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * np.sqrt(
            2 / self.rf_dim
        )

        logits = torch.einsum("bzhw,zc->bchw", phi, self.beta)

        if self.training:
            self.covariance = None
            return {
                "probability": torch.softmax(logits, dim=1),
                "semantic": logits,
                "logits": logits,
                "phi": phi,
            }
        else:
            if self.covariance is None:
                self.compute_covariance()
            var = torch.einsum("cyz,bzhw->bcyhw", self.covariance, phi)
            var = torch.einsum("bcyhw,byhw->bchw", var, phi)

            means = (
                torch.zeros_like(logits)
                .unsqueeze(-1)
                .repeat((1, 1, 1, 1, self.mc_samples))
            )

            samples = torch.normal(means, 1) * var.unsqueeze(
                -1
            ).sqrt() + logits.unsqueeze(-1)
            softmaxs = F.softmax(samples, dim=1)

            if self.uncertainty_mode == "var_max":
                uncertainty = softmaxs.var(dim=4).sqrt() * 2
                softmaxs = softmaxs.mean(dim=4)

                certainty = 1 - uncertainty

                return {
                    "probability": softmaxs,
                    "semantic": logits,
                    "logits": logits,
                    "phi": phi,
                    "var": var,
                    "certainty": certainty,
                }
            else:
                softmaxs = softmaxs.mean(dim=4)

                return {
                    "probability": softmaxs,
                    "semantic": logits,
                    "logits": logits,
                    "phi": phi,
                    "var": var,
                }

    def compute_covariance(self):
        self.covariance = torch.linalg.inv(self.precision)

    def get_uncertainty(self, out, i):
        return 1 - out["semantic_additional"]["certainty"][i].max(dim=0).values

    def training_step(self, x, out):
        with torch.no_grad():
            phi = out["semantic_additional"]["phi_small"]
            p = out["semantic_additional"]["probability_small"]

            x = p * (1 - p)
            precs = torch.einsum("bchw,bihw,bjhw->cij", x, phi, phi)

            self.precision = self.gamma * self.precision + (1 - self.gamma) * precs


class SoftmaxHead(SinglePanopticDeepLabHead):
    def __init__(
        self,
        num_classes=21,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_classes=[num_classes], class_key=["result"], *args, **kwargs
        )

    def forward(self, x):
        logits = super().forward(x)["result"]

        semantic = F.softmax(logits, dim=1)

        return {
            "semantic": semantic,
            "logits": logits,
        }

    def get_uncertainty(self, out, i):
        return 1 - out["semantic"][i].max(dim=0).values

    def training_step(self, x, out):
        pass


def get_semantic_head(tpe: str, *args, **kwargs):
    if tpe == "dpn":
        return DPNHead(*args, **kwargs)
    elif tpe == "duq":
        return DUQHead(*args, **kwargs)
    elif tpe == "sngp":
        return SNGPHead(*args, **kwargs)
    elif tpe == "softmax":
        return SoftmaxHead(*args, **kwargs)
    raise Exception(f"Semantic head called {tpe} not known")
