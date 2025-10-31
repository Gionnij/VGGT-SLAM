import torch
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    remap_instance,
)
from torch.utils.dlpack import from_dlpack, to_dlpack

# from Stream_Based_AL.utils.import_helper import get_cudf, get_cuml
# import cudf
# import cuml

EPSILON = 0.000001


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    if ctr_hmp.size(0) != 1:
        raise ValueError("Only supports inference for batch size = 1")

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(
        ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding
    )
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze(dim=0)
    assert len(ctr_hmp.size()) == 3, "Something is wrong with center heatmap dimension."

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def postprocess(
    out,
    threshold,
    nms_kernel,
    top_k_instance,
    spatial_clustering,
    dm,
    bias_weight=1.0,
    no_class_cfg=None,
    mixture_dist="normal",
    predict_sigmas=True,
    instance_class="detection",
    semantic_uncertainty=None,
    semantic_head=None,
    unnormalized_dists=False,
    certainty_mean=None,
    certainty_var=None,
    distance_mean=None,
    distance_var=None,
    certainty_threshold=None,
    distance_threshold=None,
    unknown_clustering=None,
    only_unknown=False,
    cluster_all=None,
    dirichlet_entropy=True,
    logit_entropy=True,
):
    if cluster_all is not None:
        return panoptic_from_clustering(out, dm, **cluster_all)

    ctr_hmp = out["detection"]
    device = ctr_hmp.device

    batch_size = ctr_hmp.shape[0]
    height, width = ctr_hmp.shape[2], ctr_hmp.shape[3]

    semantic = torch.zeros((batch_size, height, width), dtype=torch.long, device=device)
    instances = torch.zeros(
        (batch_size, height, width), dtype=torch.long, device=device
    )
    prototype_classes = []

    prototypes = []
    distances = []
    certainties = []

    embedding_dim = out["embedding"].shape[1]
    n_sigmas = 0 if not predict_sigmas else 2 if spatial_clustering else 1

    ctrs = []

    for i in range(batch_size):
        ctr = find_instance_center(
            ctr_hmp[i : i + 1],
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k_instance,
        )
        ctrs.append(ctr)
        instance_cls = ctr[:, 0]
        n_instances = ctr.shape[0]

        thing_prototypes = out["thing_prototypes"][i]

        instance_prototypes = torch.gather(
            thing_prototypes.contiguous().view((embedding_dim + n_sigmas, -1)),
            dim=1,
            index=(ctr[:, 1] * width + ctr[:, 2])
            .unsqueeze(0)
            .expand((embedding_dim + n_sigmas, n_instances)),
        )
        stuff_prototypes = out["stuff_prototypes"][i]

        all_prototypes = torch.cat(
            [instance_prototypes, stuff_prototypes], dim=1
        ).transpose(0, 1)

        prototypes.append(all_prototypes)

        n_prototypes = all_prototypes.shape[0]

        embeddings = out["embedding"][i]

        dists = torch.cdist(
            (
                all_prototypes[None, :, :-n_sigmas]
                if predict_sigmas
                else all_prototypes[None, :, :]
            ),
            embeddings.reshape((embedding_dim, -1)).transpose(0, 1).unsqueeze(0),
        ).squeeze()

        if predict_sigmas:
            sigmas = all_prototypes[:, embedding_dim : embedding_dim + 1]
        else:
            sigmas = torch.ones_like(
                all_prototypes[:, embedding_dim - 1 : embedding_dim]
            )
        yhat = calculate_yhat(
            dists,
            sigmas,
            mixture_dist=mixture_dist,
            bias_weight=bias_weight,
            n_sigmas=n_sigmas,
        )

        if spatial_clustering:
            yhat2 = calculate_pixel_distances(
                all_prototypes,
                ctr[:, 1:],
                embedding_dim,
                width,
                height,
                n_sigmas=n_sigmas,
                bias_weight=bias_weight,
                mixture_dist=mixture_dist,
            )
            yhat = yhat + yhat2

        if no_class_cfg is not None:
            yhat = torch.cat(
                (
                    yhat,
                    no_class_cfg["no_class_score"]
                    .reshape((1, 1))
                    .expand((1, yhat.shape[1])),
                ),
                dim=0,
            )

        if mixture_dist == "dirichlet":
            compute_dirichlet_mixture(height, width, distances, certainties, yhat)
        else:
            compute_normal_mixture(height, width, distances, certainties, yhat)

        if semantic_uncertainty is not None:
            uncertainty = semantic_head.get_uncertainty(out, i)

            certainties[i] = 1 - uncertainty

        # Captures to which prototype each pixel belongs
        prototype_correspondence = yhat.argmax(dim=0)
        if no_class_cfg is not None:
            old_correspondence = yhat[:-1].argmax(dim=0)

        if unnormalized_dists:
            if no_class_cfg is not None:
                distances[i] = dists.gather(
                    dim=0, index=old_correspondence.unsqueeze(0)
                ).reshape((height, width))
            else:
                distances[i] = dists.gather(
                    dim=0, index=prototype_correspondence.unsqueeze(0)
                ).reshape((height, width))

        # Number of prototypes + potentially the OOD prototype if necessary
        total_prototypes = yhat.shape[0]

        # here the ood_mask is calculated using the certainty statistics and distance statistic
        # if not entered here the distance to the prototypes (yhat) is used fro computing prototype correspondance
        if certainty_mean is not None and certainty_threshold is not None:
            normalized_certainties = (
                certainties[-1] - certainty_mean.to(device)
            ) / certainty_var.to(device).sqrt()

            if distance_mean is not None and distance_threshold is not None:
                if len(distance_mean.shape) > 0:
                    dists = distances[i].squeeze()

                    num_classes = out["semantic"][i].shape[0]
                    # print(num_classes)
                    pixel_classes = F.one_hot(
                        out["semantic"][i].argmax(dim=0), num_classes
                    ).reshape((-1, num_classes))
                    prototype_clss_counts = torch.einsum(
                        "np,nc->pc",
                        F.one_hot(prototype_correspondence, total_prototypes).float(),
                        pixel_classes.float(),
                    )
                    prototype_clss = prototype_clss_counts.argmax(dim=1)
                    semantic_tmp = prototype_clss.gather(
                        dim=0, index=prototype_correspondence
                    ).reshape(semantic[i].shape)

                    semantic_tmp = semantic_tmp.reshape((-1))
                    dists = dists.reshape((-1))

                    oh = F.one_hot(semantic_tmp, num_classes=num_classes)
                    normalized_dists = (
                        (dists.unsqueeze(1) * oh) - distance_mean.unsqueeze(0)
                    ) / distance_var.unsqueeze(0).sqrt()

                    normalized_dists = normalized_dists.gather(
                        dim=1, index=semantic_tmp.unsqueeze(1)
                    ).squeeze()
                    normalized_dists = normalized_dists.reshape(
                        normalized_certainties.shape
                    )
                else:
                    dists = distances[i].squeeze()
                    normalized_dists = (
                        dists - distance_mean.to(device)
                    ) / distance_var.to(device)

                ood_mask = (normalized_certainties < certainty_threshold) & (
                    normalized_dists > distance_threshold
                )
            else:
                ood_mask = normalized_certainties < certainty_threshold
                while ood_mask.sum() == 0:
                    certainty_threshold += 0.01
                    ood_mask = normalized_certainties < certainty_threshold

            old_correspondence = prototype_correspondence.clone()

            prototype_correspondence[ood_mask.reshape((-1))] = n_prototypes
            total_prototypes = total_prototypes + 1

        if no_class_cfg is not None:
            ood_mask = prototype_correspondence == (
                yhat.shape[0] - 1
            )  # meaning the no_class_score is the greatest logit
            ood_mask = ood_mask.reshape(semantic[i].shape)

        if instance_class == "majority_vote":
            num_classes = out["semantic"][i].shape[0]

            pixel_classes = F.one_hot(
                out["semantic"][i].argmax(dim=0), num_classes
            ).reshape((-1, num_classes))
            prototype_clss_counts = torch.einsum(
                "np,nc->pc",
                F.one_hot(prototype_correspondence, total_prototypes).float(),
                pixel_classes.float(),
            )
            prototype_clss = prototype_clss_counts.argmax(dim=1)

            # We have an OOD prototype, which should have no class
            if prototype_clss.shape[0] > n_prototypes:
                prototype_clss[n_prototypes] = -1

            prototype_classes.append(prototype_clss)

            semantic[i] = prototype_clss.gather(
                dim=0, index=prototype_correspondence
            ).reshape(semantic[i].shape)
        else:
            compute_class_from_prototype(
                out,
                dm,
                no_class_cfg,
                height,
                width,
                semantic,
                prototype_classes,
                i,
                instance_cls,
                n_instances,
                n_prototypes,
                prototype_correspondence,
            )

        instance = prototype_correspondence

        if unknown_clustering is not None:
            # cudf = get_cudf()
            # cuml = get_cuml()
            import cuml
            import cudf

            clusterer = cuml.cluster.DBSCAN(
                eps=unknown_clustering.get("epsilon", 0.5),
                min_samples=unknown_clustering.get("min_samples", 5),
                calc_core_sample_indices=False,
            )

            data = out["embedding"][i].detach()
            data = data[:, ood_mask]
            data = data.reshape((data.shape[0], -1)).T

            if data.shape[0] > 0:
                df = cudf.from_dlpack(to_dlpack(data))
                clusterer.fit(df)

                instance_ids = from_dlpack(clusterer.labels_.to_dlpack())
                instance_ids = remap_instance(instance_ids, -1)

                if unknown_clustering.get("reassign_outliers", False):
                    mask = instance_ids == -1

                    ood_old_correspondence = old_correspondence[ood_mask.reshape((-1))]
                    if (
                        unknown_clustering.get("greater_distance_threshold", None)
                        is not None
                    ):
                        if dists.shape[1] != len(ood_mask.reshape((-1))):
                            ood_dists = dists.reshape((-1))[ood_mask.reshape((-1))]
                        else:
                            ood_dists = dists[:, ood_mask.reshape((-1))]
                            ood_dists = ood_dists.gather(
                                dim=0, index=ood_old_correspondence.unsqueeze(0)
                            ).squeeze(0)
                        ood_dists = ood_dists[~mask.to(ood_dists.device)]

                        oh = (
                            F.one_hot(instance_ids[~mask].long())
                            .unsqueeze(2)
                            .to(device)
                        )
                        oh_data = oh * (data[~mask.to(device), :].unsqueeze(1))
                        means = oh_data.sum(dim=0) / oh.sum(dim=0)

                        new_dists = (oh_data - means).norm(dim=2)
                        new_dists = new_dists.gather(
                            dim=1,
                            index=instance_ids[~mask].to(device).unsqueeze(1).long(),
                        ).squeeze(1)

                        greater_threshold = (
                            new_dists
                            > ood_dists
                            * unknown_clustering.get("greater_distance_threshold")
                        )
                        mask[~mask] = greater_threshold.to(mask.device)

                    old = ood_old_correspondence[mask.to(device)]
                    if old.shape[0] > 0:
                        instance_ids[mask] = -old.int().to(mask.device)

                        semantic_reassign = torch.ones(
                            mask.shape, dtype=semantic.dtype, device=device
                        ) * (-1)
                        old_semantic = prototype_clss.gather(dim=0, index=old)
                        semantic_reassign[mask] = old_semantic
                        semantic[i, ood_mask] = semantic_reassign
                else:
                    # This is probably not the correct way to handle this!
                    instance_ids[instance_ids == -1] = 0

                instance[ood_mask.reshape((-1))] = -instance_ids.long().to(device)

        instance = instance.reshape((height, width))
        instances[i] = instance
    panoptic = semantic * dm.label_divisor + instances
        
    if unknown_clustering is not None:
        # print(instance)
        ##for pq evaluation of Coco
        # set predicted semantic label to 254, which is the ood class
        unknown_mask = instance * dm.label_divisor < 0
        # instances[instances < 0] = 0
        semantic[unknown_mask.unsqueeze(0)] = 254
        unknown_mask = instance * dm.label_divisor == 253
        semantic[unknown_mask.unsqueeze(0)] = 254
        semantic = semantic.squeeze().unsqueeze(0)
        


    # print(torch.unique(semantic))
    if only_unknown:
        unknown_segmentation = torch.zeros_like(panoptic)

        unknown_mask = panoptic < 0
        # unknown_segmentation[~unknown_mask] = 1

        # Unknown Panoptic ids are -1000 + instance_id and
        # have to be converted to 1000 + instance_id
        # unknown_segmentation[unknown_mask] = panoptic[unknown_mask] + 2001
        unknown_segmentation[unknown_mask] = panoptic[unknown_mask]
        unknown_segmentation = remap_instance(unknown_segmentation, 0)
        ood_mask = unknown_segmentation
        # print(unknown_segmentation.unique())
        # panoptic = unknown_segmentation
    else:
        ood_mask = torch.zeros_like(panoptic)
    panoptic = semantic * dm.label_divisor + instances
    ret = {
        "semantic": semantic,
        "panoptic": panoptic,
        "prototypes": prototypes,
        "prototype_classes": prototype_classes,
        "instance_ctrs": ctrs,
        "distances": distances,
        "certainties": torch.stack(certainties),
        "uncertainty": -torch.stack(certainties),
        "ood_mask": ood_mask,
    }
    if dirichlet_entropy:
        a = out["semantic"].permute(0, 2, 3, 1) + 1
        ret["entropy"] = Dirichlet(a).entropy()

    if logit_entropy:
        from scipy.stats import entropy as scipy_entropy

        entropy = []
        for a in out["semantic"].detach().clone():
            a += 1
            entropy.append(
                torch.from_numpy(scipy_entropy(a.cpu().detach().squeeze().numpy()))
            )

        ret["logit_entropy"] = torch.stack(entropy)

    return ret


def panoptic_from_clustering(out, dm, epsilon, min_samples, small=False, refit=False):
    batch_size = out["embedding"].shape[0]
    height = out["embedding"].shape[2]
    width = out["embedding"].shape[3]

    semantic = torch.zeros(
        (batch_size, height, width), dtype=torch.long, device=out["embedding"].device
    )
    panoptic = torch.zeros(
        (batch_size, height, width), dtype=torch.long, device=out["embedding"].device
    )

    for i in range(batch_size):
        embedding = out["embedding"][i].cpu().detach().numpy()
        if small:
            embedding = out["embedding_small"][i].cpu().detach().numpy()

        instance_ids = cluster_panoptic(
            out, i, epsilon=epsilon, min_samples=min_samples, small=small, refit=refit
        )

        instance_ids[instance_ids == -1] = instance_ids.max() + 1

        semantic_pred = out["semantic"][i].argmax(dim=0)
        instance_ids = instance_ids.reshape(embedding.shape[1:])
        if small and instance_ids.shape[0] < semantic_pred.shape[0]:
            instance_ids = (
                F.interpolate(
                    instance_ids.unsqueeze(0).unsqueeze(0).float(),
                    size=semantic_pred.shape,
                    mode="nearest",
                )
                .squeeze()
                .long()
            )
        instance_ids = instance_ids % dm.label_divisor
        instance_ids = remap_instance(instance_ids, -1)

        panoptic_id = compute_majority_vote(semantic_pred, instance_ids, out, i, dm)

        panoptic[i] = panoptic_id
        semantic[i] = semantic_pred

    return {
        "semantic": semantic,
        "panoptic": panoptic,
    }


def compute_majority_vote(semantic_pred, instance_ids, out, i, dm):
    semantic_pred = semantic_pred.cpu()
    unique_instances = instance_ids.unique()

    instance_onehot = F.one_hot(instance_ids, unique_instances.shape[0])
    semantic_onehot = F.one_hot(semantic_pred, out["semantic"][i].shape[0])
    instance_cls_counts = torch.einsum("hwi,hwc->ic", instance_onehot, semantic_onehot)

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


def cluster_panoptic(out, i, epsilon, min_samples, small=False, refit=False):
    # cuml = get_cuml()
    import cuml
    import cudf

    clusterer = cuml.DBSCAN(
        eps=epsilon,
        min_samples=min_samples,
        calc_core_sample_indices=False,
    )
    data = out["embedding_small" if small else "embedding"][i].detach()

    # cudf = get_cudf()
    data = data.reshape((data.shape[0], -1)).T
    df = cudf.from_dlpack(to_dlpack(data))

    clusterer.fit(df)
    instance_ids = from_dlpack(clusterer.labels_.to_dlpack())
    instance_ids = remap_instance(instance_ids, -1)

    if refit:
        embedding = out["embedding"][i].detach()
        embeds_small = out["embedding_small"][i].detach()

        unique, counts = instance_ids.unique(return_counts=True)
        n_instances = unique.shape[0] - 1

        if n_instances > 0:
            embed_dim = embedding.shape[0]

            instance_ids[instance_ids == -1] = n_instances
            instance_one_hot = F.one_hot(instance_ids.to(torch.int64), n_instances + 1)[
                :, :n_instances
            ]

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

    return instance_ids


def compute_class_from_prototype(
    out,
    dm,
    no_class_cfg,
    height,
    width,
    semantic,
    prototype_classes,
    i,
    instance_cls,
    n_instances,
    n_prototypes,
    prototype_correspondence,
):
    n_thing_classes = out["detection"].shape[1]
    n_stuff_classes = len(dm.mapped_stuff_list)
    # print(n_stuff_classes)
    # Classes of all prototypes, where the first n_thing_classes indices are the thing classes
    # and the remaining n_stuff_classes are the stuff classes
    prototype_clss = torch.zeros(n_prototypes, device=prototype_correspondence.device)
    prototype_clss[:n_instances] = instance_cls
    prototype_clss[n_instances:] = torch.arange(
        n_thing_classes, n_thing_classes + n_stuff_classes
    )

    # Class of each pixel

    clss_index = prototype_correspondence
    if no_class_cfg is not None:
        clss_index = clss_index.clone()
        clss_index[clss_index == n_prototypes] = 0
    clss = prototype_clss.gather(dim=0, index=clss_index)
    if no_class_cfg is not None:
        clss[prototype_correspondence == n_prototypes] = -1

    clss = clss.reshape((height, width))

    mapped_prototype_clss = torch.zeros_like(prototype_clss)

    for j, thing_id in enumerate(dm.mapped_thing_list):
        semantic[i, clss == j] = thing_id
        mapped_prototype_clss[prototype_clss == j] = thing_id
    for j, stuff_id in enumerate(dm.mapped_stuff_list):
        semantic[i, clss == j + n_thing_classes] = stuff_id
        mapped_prototype_clss[prototype_clss == j + n_thing_classes] = stuff_id

    if no_class_cfg is not None:
        semantic[i, clss == -1] = -1

    prototype_classes.append(mapped_prototype_clss)


def compute_normal_mixture(height, width, distances, certainties, yhat):
    yhat_reshaped = yhat.reshape((yhat.shape[0], height, width))
    dist = (-yhat_reshaped).min(dim=0).values
    distances.append(dist)

    certainties.append(F.softmax(yhat_reshaped, dim=0).max(dim=0).values)


def compute_dirichlet_mixture(height, width, distances, certainties, yhat):
    yhat_reshaped = yhat.reshape((yhat.shape[0], height, width))
    dist = (1 / yhat_reshaped).min(dim=0).values
    distances.append(dist)

    alpha = yhat_reshaped + 1
    alpha_sum = alpha.sum(dim=0, keepdim=False)
    uncertainty = alpha.shape[0] / alpha_sum

    certainties.append(1 - uncertainty)


def calculate_yhat(dists, sigmas, mixture_dist="normal", bias_weight=1.0, n_sigmas=1):
    embedding_dim = dists.shape[0] - n_sigmas
    if mixture_dist == "normal":
        return -dists * dists / (sigmas * 2 + EPSILON) - (
            embedding_dim / 2 * torch.log(sigmas) if bias_weight == 1 else 0.0
        )
    elif mixture_dist == "cauchy":
        return -torch.log(1 + dists * dists / (sigmas + EPSILON)) - (
            embedding_dim * torch.log(sigmas) if bias_weight == 1 else 0.0
        )
    elif mixture_dist == "dirichlet":
        if bias_weight == 1:
            raise Exception(
                "Using the dirichlet mixture distribution implies no bias weight"
            )
        return (sigmas * 2) / (dists * dists + EPSILON)
    raise Exception(f"Mixture distribution named {mixture_dist} not known")


def calculate_pixel_distances(
    all_prototypes,
    instance_positions,
    embedding_dim,
    width,
    height,
    n_sigmas,
    bias_weight=1.0,
    mixture_dist="normal",
):
    sigmas = all_prototypes[:, -n_sigmas + 1 :]
    height_pos = (
        torch.arange(0, height, 1, device=instance_positions.device)
        .reshape((height, 1, 1))
        .expand(height, width, 1)
    )
    width_pos = (
        torch.arange(0, width, 1, device=instance_positions.device)
        .reshape((1, width, 1))
        .expand(height, width, 1)
    )
    pixel_positions = torch.cat((height_pos, width_pos), dim=2).reshape((-1, 2))

    dists = torch.cdist(
        instance_positions.unsqueeze(0).float(),
        pixel_positions.unsqueeze(0).float(),
    ).squeeze(0)

    yhat = calculate_yhat(
        F.pad(dists, [0, 0, 0, sigmas.shape[0] - dists.shape[0]]),
        sigmas,
        mixture_dist=mixture_dist,
        bias_weight=bias_weight,
        n_sigmas=n_sigmas,
    )

    return yhat
