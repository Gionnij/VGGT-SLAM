import itertools
import numpy as np
from pytorch_lightning import LightningModule
import torch
from tqdm import tqdm
import torch.nn.functional as F


def remap(instance, ignore_instance=[]):
    if isinstance(instance, np.ndarray):
        instance = torch.from_numpy(instance)
    unique_instance = instance.unique()
    instance_clone = instance.clone()
    c = 1
    for i, x in enumerate(unique_instance):
        if ignore_instance is None or x.item() not in ignore_instance:
            instance_clone[instance == x] = c
            c = c + 1
    return instance_clone


def distance_stats_U3HS(pl_module: LightningModule, iters=100) -> None:
    pl_module.dm.batch_size = 1
    train_dl = pl_module.dm.train_dataloader()

    def data_loader():
        data_iter = iter(train_dl)

        for num_iter in itertools.count(1):
            batch = next(data_iter)
            x = pl_module.dm.get_image_from_batch(batch)

            # This will likely not work in multi-GPU training!
            yield x.to(pl_module.device), pl_module.dm.get_semantic_from_batch(
                batch
            ).to(pl_module.device)

    iters = len(train_dl) if iters == -1 else min(iters, len(train_dl))

    print("Computing distance stats")

    data_iter = data_loader()
    sum_c = 0
    counts = 0
    for inputs, y in tqdm(
        itertools.islice(data_iter, iters),
        total=iters,
    ):
        output = pl_module(inputs)

        postprocessed = pl_module.model.postprocess(
            output, pl_module.dm, pl_module.hparams["post_process"]
        )

        distances = postprocessed["distances"][0]
        pred_panoptic = postprocessed["panoptic"].squeeze()

        semantic = pred_panoptic // pl_module.dm.label_divisor

        mask = (y != 255).squeeze()
        distances = distances[mask]
        semantic = semantic[mask]

        sum_c += distances.sum()
        counts += semantic.shape[0]

    mean = sum_c / counts

    data_iter = data_loader()
    sum_v = 0
    for inputs, y in tqdm(
        itertools.islice(data_iter, iters),
        total=iters,
    ):
        output = pl_module(inputs)

        postprocessed = pl_module.model.postprocess(
            output, pl_module.dm, pl_module.hparams["post_process"]
        )

        distances = postprocessed["distances"][0]
        pred_panoptic = postprocessed["panoptic"].squeeze()

        semantic = pred_panoptic // pl_module.dm.label_divisor

        mask = (y != 255).squeeze()

        distances = distances[mask]
        semantic = semantic[mask]

        diff = distances - mean
        diff = diff * diff
        sum_v += diff.sum()

    variance = sum_v / counts

    pl_module.model.set_distance_stats(mean, variance)
    return mean, variance


def uncertainty_stats_m2p(pl_module, iters=-1):
    # pl_module.dm.base_size_val = (512, 256)
    data_iter = pl_module.dm.val_dataloader()

    iters = len(data_iter) if iters == -1 else min(iters, len(data_iter))

    print("Computing uncertainty stats")

    sum_c = 0
    counts = 0
    all_uncertainties = []
    for batch in tqdm(
        itertools.islice(data_iter, iters),
        total=iters,
    ):

        out = pl_module(batch["image"].cuda())
        postprocess = pl_module.model.postprocess(
            out, pl_module.dm, pl_module.hparams["post_process"]
        )
        mask = batch["semantic"] != 255
        uncertainty = postprocess["uncertainty"].cpu().numpy()[mask]
        all_uncertainties.append(uncertainty)
        del uncertainty, postprocess, out, mask, batch
    uncertainty = np.concatenate(all_uncertainties)
    mean = uncertainty.mean()
    var = uncertainty.std()
    print(mean, var)
    return mean, var


def certainty_stats_U3HS(pl_module, iters=-1):
    train_dl = pl_module.dm.train_dataloader()

    def data_loader():
        data_iter = iter(train_dl)

        for num_iter in itertools.count(1):
            batch = next(data_iter)
            x = pl_module.dm.get_image_from_batch(batch)

            # This will likely not work in multi-GPU training!
            yield x.to(pl_module.device), pl_module.dm.get_semantic_from_batch(
                batch
            ).to(pl_module.device)

    iters = len(train_dl) if iters == -1 else min(iters, len(train_dl))

    print("Computing uncertainty stats")

    data_iter = data_loader()
    sum_c = 0
    counts = 0
    all_certainties = []
    for inputs, y in tqdm(
        itertools.islice(data_iter, iters),
        total=iters,
    ):
        output = pl_module(inputs)
        certainties = []
        for i in range(output["semantic"].shape[0]):
            uncertainty = pl_module.model.decoder.semantic_head.get_uncertainty(
                output, i
            )

            certainties.append(1 - uncertainty)

        certainties = torch.stack(certainties)

        mask = y != 255
        certainties = certainties[mask]
        all_certainties.append(certainties)
        sum_c += certainties.sum()
        counts += mask.sum()
        del certainties, mask, y, inputs, output
    mean = sum_c / counts
    all_certainties = torch.cat(all_certainties, dim=0)
    diff = all_certainties - mean

    diff = diff * diff

    sum_v = diff.sum()
    variance = sum_v / counts

    pl_module.model.set_certainty_stats(mean, variance)
    print(f"Mean: {mean}, Variance: {variance}")

    del sum_v, sum_c
    torch.cuda.empty_cache()
    return mean, variance
