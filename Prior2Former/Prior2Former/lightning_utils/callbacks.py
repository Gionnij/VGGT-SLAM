import itertools
from typing import Dict

import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
from fvcore.nn.precise_bn import update_bn_stats
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule

# from lightning_utils.core.lightning import LightningModule
from metrics.streamIoU import StreamSegMetrics


class OptunaCallback(Callback):
    def __init__(self, trial: optuna.Trial, metric: str = "val_iou"):
        super().__init__()

        self.trial = trial
        self.metric = metric

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metric = trainer.logged_metrics[self.metric]

        self.trial.report(metric, trainer.current_epoch)

        if self.trial.should_prune():
            print(f"Pruning trial {self.trial.number} with metric {metric}")
            raise optuna.TrialPruned()


class GradientNormMonitor(Callback):
    r"""
    Automatically monitor and logs gradient norms of the model.
    """

    def __init__(self):
        super().__init__()

    def on_after_backward(self, trainer, pl_module: LightningModule) -> None:
        # def on_before_zero_grad(
        # self, optimizer, trainer, pl_module: LightningModule
        # ) -> None:
        """Called after ``loss.backward()`` and before optimizers do anything."""
        # trainer.logger.log_metrics(latest_stat, step=trainer.global_step)
        return None
        grad_norms = torch.stack(
            [x.grad.norm(p=2) for x in pl_module.parameters() if x.grad is not None]
        )

        trainer.logger.log_metrics(
            {
                "grad/min": grad_norms.min().item(),
                "grad/max": grad_norms.max().item(),
                "grad/mean": grad_norms.mean().item(),
                "grad/median": grad_norms.median().item(),
            },
            step=trainer.global_step,
        )
        if len(trainer.loggers) == 2:
            trainer.loggers[1].log_metrics(
                {
                    "grad/min": grad_norms.min().item(),
                    "grad/max": grad_norms.max().item(),
                    "grad/mean": grad_norms.mean().item(),
                    "grad/median": grad_norms.median().item(),
                },
                step=trainer.global_step,
            )


class ChangeOutputStride(Callback):
    def __init__(
        self,
        os: int,
        epoch: int,
        compute_exact_bn=True,
        exact_bn_iter=200,
        freeze_bn=True,
        lower_batch_size=None,
    ):
        super().__init__()

        self.os = os
        self.epoch = epoch
        self.compute_exact_bn = compute_exact_bn
        self.exact_bn_iter = exact_bn_iter
        self.freeze_bn = freeze_bn
        self.lower_batch_size = lower_batch_size

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: LightningModule
    ) -> None:
        if self.freeze_bn and self.epoch <= trainer.current_epoch:
            count = 0
            for m in pl_module.model.modules():
                if isinstance(
                    m,
                    (
                        torch.nn.BatchNorm1d,
                        torch.nn.BatchNorm2d,
                        torch.nn.BatchNorm3d,
                        torch.nn.SyncBatchNorm,
                    ),
                ):
                    m.eval()
                    count += 1
            print(f"Froze {count} BN modules")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: LightningModule
    ) -> None:
        if self.epoch == trainer.current_epoch + 1:
            pl_module.model.set_output_stride(self.os)
            pl_module.model.to(pl_module.device)

            trainer.optimizers = [pl_module.get_optimizer()]

            if self.lower_batch_size is not None:
                pl_module.dm.batch_size = self.lower_batch_size

            if self.compute_exact_bn:
                print("Updating BN stats")

                def data_loader():
                    data_iter = iter(pl_module.dm.train_dataloader())

                    for num_iter in itertools.count(1):
                        batch = next(data_iter)
                        x = pl_module.dm.get_image_from_batch(batch)

                        # This will likely not work in multi-GPU training!
                        yield x.to(pl_module.device)

                with torch.no_grad():
                    update_bn_stats(
                        pl_module.model, data_loader(), self.exact_bn_iter, "tqdm"
                    )


class ComputeLogitStats(Callback):
    def __init__(
        self,
        iters=-1,
    ):
        super().__init__()

        self.iters = iters

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        train_dl = pl_module.dm.train_dataloader()

        def data_loader():
            data_iter = iter(train_dl)

            for num_iter in itertools.count(1):
                batch = next(data_iter)
                x = pl_module.dm.get_image_from_batch(batch)

                # This will likely not work in multi-GPU training!
                yield x.to(pl_module.device)

        iters = len(train_dl) if self.iters == -1 else min(self.iters, len(train_dl))

        print("Computing logit mean")

        data_iter = data_loader()
        sum_logits = 0
        counts = 0
        for inputs in tqdm.tqdm(
            itertools.islice(data_iter, iters),
            total=iters,
        ):
            output = pl_module(inputs)
            logits = output["prediction"]
            pred = logits.argmax(dim=1)

            pred_oh = F.one_hot(pred, num_classes=logits.shape[1]).permute((0, 3, 1, 2))

            counts = counts + pred_oh.sum(dim=(0, 2, 3))
            sum_logits = sum_logits + (pred_oh * logits).sum(dim=(0, 2, 3))

        mean = sum_logits / counts

        print("Computing logit variance")

        data_iter = data_loader()
        diffs = 0
        for inputs in tqdm.tqdm(
            itertools.islice(data_iter, iters),
            total=iters,
        ):
            output = pl_module(inputs)
            logits = output["prediction"]
            pred = logits.argmax(dim=1)
            pred_oh = F.one_hot(pred, num_classes=logits.shape[1]).permute((0, 3, 1, 2))

            diff = logits - mean.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            diff = diff * diff
            diffs = diffs + (pred_oh * diff).sum(dim=(0, 2, 3))

        variance = diffs / counts

        pl_module.model.classifier.set_class_mean_var(mean, variance)


class ComputeUncertaintyStats(Callback):
    def __init__(self, iters=-1, before_test=False):
        super().__init__()

        self.iters = iters
        self.before_test = before_test

    def on_test_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        if self.before_test:
            self.on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
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

        iters = len(train_dl) if self.iters == -1 else min(self.iters, len(train_dl))

        print("Computing uncertainty stats")

        data_iter = data_loader()
        sum_c = 0
        counts = 0
        all_certainties = []
        for inputs, y in tqdm.tqdm(
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

            # postprocessed = pl_module.model.postprocess(
            #     output, pl_module.dm, pl_module.hparams["post_process"]
            # )

            # certainties = postprocessed["certainties"]
            certainties = torch.stack(certainties)

            mask = y != 255
            certainties = certainties[mask]
            all_certainties.append(certainties)
            sum_c += certainties.sum()
            counts += mask.sum()

        mean = sum_c / counts
        all_certainties = torch.cat(all_certainties, dim=0)
        diff = all_certainties - mean
        # data_iter = data_loader()
        sum_v = diff * diff
        # for inputs, y in tqdm.tqdm(
        #     itertools.islice(data_iter, iters),
        #     total=iters,
        # ):
        #     output = pl_module(inputs)

        #     postprocessed = pl_module.model.postprocess(
        #         output, pl_module.dm, pl_module.hparams["post_process"]
        #     )

        #     certainties = postprocessed["certainties"]
        #     certainties = torch.stack(certainties)

        #     mask = y != 255
        #     certainties = certainties[mask]

        #     diff = certainties - mean
        #     diff = diff * diff

        sum_v = diff.sum()
        variance = sum_v / counts

        pl_module.model.set_certainty_stats(mean, variance)

        del variance, mean, sum_v, sum_c
        torch.cuda.empty_cache()


class ComputeSemSegMetrics(Callback):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        train_dl = pl_module.dm.train_dataloader()
        val_dl = pl_module.dm.val_dataloader()

        sm_train = StreamSegMetrics(pl_module.dm.num_classes)
        sm_val = StreamSegMetrics(pl_module.dm.num_classes)

        def data_loader(dl):
            data_iter = iter(dl)

            for num_iter in itertools.count(1):
                batch = next(data_iter)
                x = pl_module.dm.get_image_from_batch(batch)
                y = pl_module.dm.get_semantic_from_batch(batch)

                # This will likely not work in multi-GPU training!
                yield x.to(pl_module.device), y.to(pl_module.device)

        iters_train = len(train_dl)
        iters_val = len(val_dl)

        # training data
        data_iter_train = data_loader(train_dl)
        loss_train = []

        for inputs, labels in tqdm.tqdm(
            itertools.islice(data_iter_train, iters_train),
            total=iters_train,
        ):
            output = pl_module(inputs)
            preds = output["semantic"]
            loss = self.ce_loss(preds.cpu(), labels.cpu())
            loss_train.append(loss.detach().data.cpu().numpy())

            sm_train.update(
                labels.cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy()
            )  # .max(): tuple (max, argmax). So [1]

        train_results = sm_train.get_results()

        # validation_data
        data_iter_val = data_loader(val_dl)
        loss_val = []

        for inputs, labels in tqdm.tqdm(
            itertools.islice(data_iter_val, iters_val),
            total=iters_val,
        ):
            output = pl_module(inputs)
            preds = output["semantic"]
            loss = self.ce_loss(preds.cpu(), labels.cpu())
            loss_val.append(loss.detach().data.cpu().numpy())

            sm_val.update(
                labels.cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy()
            )

        val_results = sm_val.get_results()

        train_metric = {
            "mIoU": train_results["Mean IoU"],
            "MIoU": train_results["Micro Mean IoU"],
            "cmIoU": train_results["Mean Class IoU"],
        }
        val_metric = {
            "mIoU": val_results["Mean IoU"],
            "MIoU": val_results["Micro Mean IoU"],
            "cmIoU": val_results["Mean Class IoU"],
        }

        train_loss = sum(loss_train) / len(loss_train)
        val_loss = sum(loss_val) / len(loss_val)

        train_stats = {"metric": train_metric, "loss": train_loss}
        val_stats = {"metric": val_metric, "loss": val_loss}

        pl_module.model.set_semseg_metrics(train_stats, val_stats)


class ComputeSemSegTestMetrics(Callback):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:

        test_dl = pl_module.dm.test_dataloader()
        sm_test = StreamSegMetrics(pl_module.dm.num_classes)

        def data_loader(dl):
            data_iter = iter(dl)

            for num_iter in itertools.count(1):
                batch = next(data_iter)
                x = pl_module.dm.get_image_from_batch(batch)
                y = pl_module.dm.get_semantic_from_batch(batch)

                # This will likely not work in multi-GPU training!
                yield x.to(pl_module.device), y.to(pl_module.device)

        iters_test = len(test_dl)

        # test data
        data_iter_test = data_loader(test_dl)
        loss_test = []

        for inputs, labels in tqdm.tqdm(
            itertools.islice(data_iter_test, iters_test),
            total=iters_test,
        ):
            output = pl_module(inputs)
            preds = output["semantic"]
            loss = self.ce_loss(preds.cpu(), labels.cpu())
            loss_test.append(loss.detach().data.cpu().numpy())

            sm_test.update(
                labels.cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy()
            )  # .max(): tuple (max, argmax). So [1]

        test_results = sm_test.get_results()
        test_metric = {
            "mIoU": test_results["Mean IoU"],
            "MIoU": test_results["Micro Mean IoU"],
            "cmIoU": test_results["Mean Class IoU"],
        }
        test_loss = sum(loss_test) / len(loss_test)
        test_stats = {"metric": test_metric, "loss": test_loss}

        pl_module.model.set_semseg_test_metrics(test_stats)


class ComputeDistanceStats(Callback):
    def __init__(self, iters=-1, before_test=False, per_class=False):
        super().__init__()

        self.iters = iters
        self.before_test = before_test
        self.per_class = per_class

    def on_test_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        if self.before_test:
            self.on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
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

        iters = len(train_dl) if self.iters == -1 else min(self.iters, len(train_dl))

        print("Computing uncertainty stats")

        data_iter = data_loader()
        sum_c = 0
        counts = 0
        for inputs, y in tqdm.tqdm(
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

            if not self.per_class:
                sum_c += distances.sum()
                counts += semantic.shape[0]
            else:
                oh = F.one_hot(semantic, num_classes=pl_module.dm.num_classes)
                sum_c += (oh * distances.unsqueeze(1)).sum(dim=0)
                counts += oh.sum(dim=0)

        mean = sum_c / counts

        data_iter = data_loader()
        sum_v = 0
        for inputs, y in tqdm.tqdm(
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

            if not self.per_class:
                diff = distances - mean
                diff = diff * diff
                sum_v += diff.sum()
            else:
                oh = F.one_hot(semantic, num_classes=pl_module.dm.num_classes)
                vals = oh * distances.unsqueeze(1)
                diffs = vals - mean.unsqueeze(0)
                diffs = diffs * diffs
                sum_v += diffs.sum(dim=0)

        variance = sum_v / counts

        pl_module.model.set_distance_stats(mean, variance)


def get_callback(name: str, args: Dict):
    if name == "change_output_stride":
        return ChangeOutputStride(**args)
    elif name == "compute_logit_statistics":
        return ComputeLogitStats(**args)
    elif name == "compute_uncertainty_statistics":
        return ComputeUncertaintyStats(**args)
    elif name == "compute_distance_statistics":
        return ComputeDistanceStats(**args)
    raise Exception(f"Callback named {name} unknown")
