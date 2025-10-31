import traceback
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from mlflow.tracking import MlflowClient
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from Benchmarks.Open_World_Benchmark.estimators import get_estimator
from Data_Loaders.lightning_data_modules import get_datamodule
from metrics.iou_confusion import IoUConfusion
from models.DeepLabV3Base.pl_model_utils import get_model

from Pytorch_Extentions.scheduler import SeqLR, PolyLR
from torch.optim.lr_scheduler import (
    MultiStepLR,
    PolynomialLR,
)

from metrics import get_metric
from metrics.pq_u3hs import PanopticQuality
from Visualizations.util import plot_confusion_matrix

# from settings import EXPERIMENTS_PATH
import os
import shutil
from utils.logging_utils.log_writers import (
    gl_error,
)


class PanopticSegmentationModule(pl.LightningModule):

    def __init__(
        self,
        model: Dict,
        log_folder=os.path.join("dummy"),
        optimizer: Dict = {},
        loss: Dict = {},
        datamodule: Dict = {},
        post_process: Dict = {},
        test_metrics: Dict = {},
        val_metrics: Dict = {},
        max_validation_imgs=8,
        max_test_imgs=8,
        plotting={},
        pq_args={},
        logger=None,
        debug: bool = False,
        full_frequency: int = 1,
        small_validation_size: int = 100,
    ):
        super().__init__()
        self.debug = debug
        self.full_frequency = full_frequency
        self.small_validation_size = small_validation_size
        self.save_hyperparameters(
            "model",
            "optimizer",
            "datamodule",
            "post_process",
            "test_metrics",
            "plotting",
        )
        if "data_dir" in datamodule["args"]:
            self.dm = get_datamodule(
                datamodule["name"], datamodule["args"], datamodule["args"]["data_dir"]
            )
        elif datamodule["name"] == "cityscapes":
            self.dm = get_datamodule(
                datamodule["name"],
                datamodule["args"],
                os.path.expanduser("~/Datasets/Cityscapes"),
            )

        self.num_classes = self.dm.num_classes
        exp_dir = os.environ.get(
            "EXPERIMENTS", os.path.join(os.path.expanduser("~"), "Experiments")
        )
        self.log_folder = os.path.join(exp_dir, log_folder, "artifacts")
        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(os.path.join(self.log_folder, "current"), exist_ok=True)
        model["args"]["num_classes"] = self.num_classes
        self.has_autoencoder = "autoencoder" in model["name"]
        if "freeze" in model["backbone"]:
            self.freeze_backbone = model["backbone"].pop("freeze")
            model["backbone"]
        else:
            self.freeze_backbone = -1
        self.model = get_model(args=model, dm=self.dm)
        if "bn_momentum_backbone" in model:
            if hasattr(self.model, "backbone"):
                self.set_bn_momentum(
                    self.model.get_backbone_modules(), model["bn_momentum_backbone"]
                )
        if "bn_momentum_decoder" in model:
            if hasattr(self.model, "decoder"):
                self.set_bn_momentum([self.model.decoder], model["bn_momentum_decoder"])
            if hasattr(self.model, "sem_seg_head"):
                self.set_bn_momentum(
                    [self.model.sem_seg_head], model["bn_momentum_decoder"]
                )
        self.name = model["name"]
        self.best = -1
        self.num_validation_imgs = 0
        self.max_validation_imgs = max_validation_imgs
        self.num_test_imgs = 0
        self.max_test_imgs = max_test_imgs
        self.val_step = 0
        self.train_step = 0

        self.iou_confusion = IoUConfusion(
            self.num_classes + 1, self.num_classes, compute_on_step=False
        )
        self.pq = PanopticQuality(
            self.num_classes,
            label_divisor=self.dm.label_divisor,
            compute_on_step=False,
            **pq_args,
        )

        self.test_metrics = {}
        for metric_name in test_metrics:
            self.test_metrics[metric_name] = get_metric(**test_metrics[metric_name])

        self.val_metrics = {}
        for metric_name in val_metrics:
            self.val_metrics[metric_name] = get_metric(**val_metrics[metric_name])

        self.plotting = plotting

    def set_bn_momentum(self, modules, momentum=0.1):
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum

    def set_estimator(self, name, args):
        self.estimator = get_estimator(name, args)

    def get_latent(self, x):
        return self.model.get_latent(x)

    def forward(self, x):
        if isinstance(x, dict):
            x = self.dm.get_image_from_batch(x).to(self.device)
        x = self.model(x)
        return x

    def on_fit_start(self):
        if hasattr(self.logger, "run_id"):
            self.model.run_id = self.logger.run_id

    def on_train_epoch_start(self):
        if self.has_autoencoder and self.current_epoch == self.model.autoencoder.freeze:
            print(
                "------------------------------freezing autoencoder-----------------------"
            )
            for param in self.model.autoencoder.parameters():
                param.requires_grad = False

        if self.freeze_backbone >= 0 and self.freeze_backbone == self.current_epoch:
            # else loaded checkpoint does not freeze the backbone
            print(
                "-----------------------freezing backbone-----------------------------"
            )
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):

        x = self.dm.get_image_from_batch(batch)

        out = self(x)

        self.model.training_step(batch, out)

        losses = self.model.loss(
            out,
            batch,
            step=self.global_step,
            epoch=self.current_epoch,
            training=True,
            dm=self.dm,
        )

        for k in losses.keys():
            try:
                self.trainer.logger.log_metrics(
                    {
                        f"train_loss/{k}": (
                            losses[k].detach().item()
                            if hasattr(losses[k], "detach")
                            else losses[k].item()
                        )
                    },
                    step=self.train_step,
                )
                if len(self.trainer.loggers) == 2:
                    self.trainer.loggers[1].log_metrics(
                        {
                            f"train_loss/{k}": (
                                losses[k].detach().item()
                                if hasattr(losses[k], "detach")
                                else losses[k].item()
                            )
                        },
                        step=self.train_step,
                    )
            except Exception as e:
                gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")

        if not losses["total"].detach().cpu().isnan().item():
            return losses["total"]
        else:
            print("Loss is NaN!")

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.train_step += 1

    def on_train_epoch_end(self):
        self.num_validation_imgs = 0
        checkpoint_path = (
            f"{self.trainer.checkpoint_callback.dirpath}/final_checkpoint.ckpt"
        )
        self.trainer.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def on_validation_epoch_start(self):
        self.total_val_loss = []

    def validation_step(self, batch, batch_idx):
        if (
            self.current_epoch + 1
        ) % self.full_frequency != 0 and batch_idx > self.small_validation_size:
            return None
        x = self.dm.get_image_from_batch(batch)

        out = self(x)

        losses = self.model.loss(
            out,
            batch,
            step=self.global_step,
            epoch=self.current_epoch,
            training=False,
            dm=self.dm,
        )
        self.total_val_loss.append(losses["total"].detach().cpu().item())
        for k in losses.keys():
            try:
                self.trainer.logger.log_metrics(
                    {
                        f"val_loss/{k}": (
                            losses[k].detach().item()
                            if hasattr(losses[k], "detach")
                            else losses[k].item()
                        )
                    },
                    step=self.val_step,
                )
                if len(self.trainer.loggers) == 2:
                    self.trainer.loggers[1].log_metrics(
                        {
                            f"val_loss/{k}": (
                                losses[k].detach().item()
                                if hasattr(losses[k], "detach")
                                else losses[k].item()
                            )
                        },
                        step=self.val_step,
                    )
            except Exception as e:
                gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")

        batch_size = batch["semantic"].shape[0]

        postprocessed = self.model.postprocess(
            out, self.dm, self.hparams["post_process"]
        )

        y_iou = batch["semantic"].clone()
        y_iou[y_iou == 255] = self.num_classes  # set to ignore index

        semantic = postprocessed["semantic"]
        if (semantic == -1).sum().cpu().item() > 0:
            semantic = semantic.clone()
            semantic[semantic == -1] = self.num_classes
        self.iou_confusion(semantic, y_iou)
        if "panoptic" in postprocessed:
            for i in range(batch_size):
                self.pq(
                    postprocessed["panoptic"][i].clone(),
                    batch["panoptic_id"][i],
                )

        for val_metric_name in self.val_metrics:
            val_metric = self.val_metrics[val_metric_name]
            val_metric(batch, out, postprocessed)

        if self.num_validation_imgs < self.max_validation_imgs:
            if (
                "ood" not in batch
                or (not "ood_threshold" in self.plotting)
                or batch["ood"].sum().cpu().item() >= self.plotting["ood_threshold"]
            ):
                grid_pred = self.model.plot_prediction(
                    x,
                    batch,
                    postprocessed,
                    out,
                    self.dm,
                    experiment=self.logger.experiment,
                    batch_idx=self.num_validation_imgs,
                    global_step=self.val_step,
                    log_folder=self.log_folder,
                )

                self.num_validation_imgs = self.num_validation_imgs + batch_size

        return self.model.collect_set_data(batch, out, postprocessed, self.dm)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.val_step += 1

    def on_validation_epoch_end(self, valset_data=None, *args, **kwargs):
        iou, confusion_matrix = self.iou_confusion.compute()
        self.iou_confusion.reset()
        confusion_matrix = (
            confusion_matrix[: self.num_classes, : self.num_classes].cpu().numpy()
        )
        img_confusion = plot_confusion_matrix(
            confusion_matrix, self.dm.get_classnames()
        )
        try:
            self.log("val_iou", iou, sync_dist=True)
        except Exception as e:
            gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")
        if isinstance(self.logger.experiment, MlflowClient):
            img = Image.fromarray(
                (img_confusion.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            img.save(os.path.join(self.log_folder, f"current/confusion-matrix.png"))
            # self.logger.experiment.log_image(
            #     run_id=self.model.run_id,
            #     image=img,
            #     artifact_file=f"current/confusion-matrix.png",
            # )
        else:
            self.logger.experiment.add_image(
                "confusion-matrix", img_confusion, self.global_step
            )

        pq, sq, rq, tp, fp, fn = self.pq.compute()
        self.pq.reset()
        try:
            self.log("mean_val_loss", torch.tensor(self.total_val_loss).mean())
            self.log("validation_pq", pq, sync_dist=True)
            self.log("val_pq/sq", sq, sync_dist=True)
            self.log("val_pq/rq", rq, sync_dist=True)
            self.log("val_pq/tp", tp, sync_dist=True)
            self.log("val_pq/fp", fp, sync_dist=True)
            self.log("val_pq/fn", fn, sync_dist=True)
        except Exception as e:
            gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")

        if self.trainer.checkpoint_callback.monitor == "validation_pq":
            self.copy_best_artifacts(metric=-pq.cpu().item(), best=-self.best)
        elif self.trainer.checkpoint_callback.monitor == "val_iou":
            self.copy_best_artifacts(metric=-iou.cpu().item(), best=-self.best)
        elif self.trainer.checkpoint_callback.monitor == "mean_val_loss":
            self.copy_best_artifacts(
                metric=torch.tensor(self.total_val_loss).mean().cpu(), best=self.best
            )

        for val_metric_name in self.val_metrics:
            val_metric = self.val_metrics[val_metric_name]

            try:
                result = val_metric.compute()

                if isinstance(result, tuple):
                    names = val_metric.get_return_names()
                    for res, name in zip(result, names):
                        try:
                            self.log(
                                f"val_{val_metric_name}/{name}", res, sync_dist=True
                            )
                        except Exception as e:
                            gl_error(
                                f"Got error: {str(e)} with trace {traceback.format_exc()}"
                            )
                elif isinstance(result, dict):
                    for name in result:
                        try:
                            self.log(
                                f"val_{val_metric_name}/{name}",
                                result[name],
                                sync_dist=True,
                            )
                        except Exception as e:
                            gl_error(
                                f"Got error: {str(e)} with trace {traceback.format_exc()}"
                            )
                else:
                    try:
                        self.log(f"test_{val_metric_name}", result, sync_dist=True)
                    except Exception as e:
                        gl_error(
                            f"Got error: {str(e)} with trace {traceback.format_exc()}"
                        )
            except ValueError as e:
                if (
                    e.args[0]
                    == "No positive samples in targets, true positive value should be meaningless"
                ):
                    print("No positive samples")
                else:
                    raise e
            val_metric.reset()
        if valset_data is not None:
            self.model.process_set_data(
                valset_data, experiment=self.logger.experiment, dm=self.dm
            )

    def test_step(self, batch, batch_idx):
        batch_size = batch["image"].shape[0]

        x = self.dm.get_image_from_batch(batch)

        out = self(x)

        postprocessed = self.model.postprocess(
            out, self.dm, self.hparams["post_process"]
        )

        for test_metric_name in self.test_metrics:
            test_metric = self.test_metrics[test_metric_name]
            test_metric(batch, out, postprocessed)

        if self.num_test_imgs < self.max_test_imgs:
            ood_mask = self.dm.get_ood_mask(batch)
            if (
                ood_mask is None
                or (not "ood_threshold" in self.plotting)
                or ood_mask.sum().cpu().item() >= self.plotting["ood_threshold"]
            ):
                val_ret = self.model.plot_prediction(
                    x,
                    batch,
                    postprocessed,
                    out,
                    self.dm,
                    log_folder=self.log_folder,
                    experiment=self.logger.experiment,
                    batch_idx=self.num_test_imgs,
                    global_step=self.global_step,
                )

                self.num_test_imgs = self.num_test_imgs + batch_size

        return self.model.collect_set_data(batch, out, postprocessed, self.dm)

    def test_epoch_end(self, testset_data, *args, **kwargs) -> None:
        for test_metric_name in self.test_metrics:
            test_metric = self.test_metrics[test_metric_name]

            result = test_metric.compute()
            if isinstance(result, tuple):
                names = test_metric.get_return_names()
                for res, name in zip(result, names):
                    try:
                        self.log(f"test_{test_metric_name}/{name}", res, sync_dist=True)
                    except Exception as e:
                        gl_error(
                            f"Got error: {str(e)} with trace {traceback.format_exc()}"
                        )
            else:
                try:
                    self.log(f"test_{test_metric_name}", result, sync_dist=True)
                except Exception as e:
                    gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")

        self.model.process_set_data(
            testset_data, experiment=self.logger.experiment, dm=self.dm
        )

    def get_lr(self):
        lr = 1e-3
        if "lr" in self.hparams["optimizer"]:
            lr = self.hparams["optimizer"]["lr"]
        elif "lr" in self.hparams["optimizer"]["args"]:
            lr = self.hparams["optimizer"]["args"]["lr"]

        if "lr_scale" in self.hparams["optimizer"]:
            lr = lr * self.hparams["optimizer"]["lr_scale"]

        return lr

    def get_lr_scheduler(self, options, optimizer):
        if options["type"] == "step":
            return optim.lr_scheduler.StepLR(optimizer=optimizer, **options["args"])
        if options["type"] == "adjust":
            return SeqLR(optimizer=optimizer, **options["args"])
        if options["type"] == "poly":
            return PolyLR(optimizer=optimizer, **options["args"])
        if options["type"] == "polynomial":
            return PolynomialLR(optimizer=optimizer, **options["args"])
        if options["type"] == "multistep":
            return MultiStepLR(optimizer=optimizer, **options["args"])
        raise Exception(f'Unknown lr scheduler type {options["type"]}')

    def has_lr_scheduler(self):
        return "lr_scheduler" in self.hparams["optimizer"]

    def get_model_parameter_dict(self):
        if self.name == "mask2former":
            return self.model.get_model_parameter_dict(
                self.hparams["optimizer"],
            )
        elif self.name == "u3hs" or self.name == "panoptic_deeplab":
            params = self.parameters()
            if "backbone_lr_factor" in self.hparams["optimizer"]:
                backbone_params = self.model.get_backbone_params()

                backbone_param_names = set([name for name, param in backbone_params])

                params = [
                    {
                        "params": [param for name, param in backbone_params],
                        "lr": self.get_lr()
                        * self.hparams["optimizer"]["backbone_lr_factor"],
                    },
                    {
                        "params": [
                            param
                            for name, param in self.model.named_parameters()
                            if name not in backbone_param_names
                        ],
                        "lr": self.get_lr(),
                    },
                ]
            return params
        else:
            raise NotImplementedError(
                f"The parameter_dict is not implemented for >>{self.name}<<"
            )

    def get_optimizer(self):
        params = self.get_model_parameter_dict()

        if self.hparams["optimizer"]["type"] == "adam":
            return optim.Adam(params, **self.hparams["optimizer"]["args"])
        elif self.hparams["optimizer"]["type"] == "sgd":
            return optim.SGD(params, **self.hparams["optimizer"]["args"])
        elif self.hparams["optimizer"]["type"] == "adamw":
            return optim.AdamW(params, **self.hparams["optimizer"]["args"])
        raise Exception(f'Unknown optimizer type {self.hparams["optimizer"]["type"]}')

    def configure_optimizers(self):
        optimizer = self.get_optimizer()

        if "lr_scheduler" in self.hparams["optimizer"]:
            lr_scheduler = self.get_lr_scheduler(
                self.hparams["optimizer"]["lr_scheduler"], optimizer
            )

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return optimizer

    # def on_before_zero_grad(self, optimizer):
    #     grad_norms = torch.stack(
    #         [x.grad.norm(p=2) for x in self.parameters() if x.grad is not None]
    #     )

    #     self.trainer.logger.log_metrics(
    #         {
    #             "grad/min": grad_norms.min().item(),
    #             "grad/max": grad_norms.max().item(),
    #             "grad/mean": grad_norms.mean().item(),
    #             "grad/median": grad_norms.median().item(),
    #         },
    #         step=self.trainer.global_step,
    #     )
    #     if len(self.trainer.loggers) == 2:
    #         self.trainer.loggers[1].log_metrics(
    #             {
    #                 "grad/min": grad_norms.min().item(),
    #                 "grad/max": grad_norms.max().item(),
    #                 "grad/mean": grad_norms.mean().item(),
    #                 "grad/median": grad_norms.median().item(),
    #             },
    #             step=self.trainer.global_step,
    #         )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        grad_norms = [x.grad.norm(p=2) for x in self.parameters() if x.grad is not None]
        if len(grad_norms) > 0:
            grad_norms = torch.stack(grad_norms)

            self.trainer.logger.log_metrics(
                {
                    "grad/min": grad_norms.min().item(),
                    "grad/max": grad_norms.max().item(),
                    "grad/mean": grad_norms.mean().item(),
                    "grad/median": grad_norms.median().item(),
                },
                step=self.trainer.global_step,
            )
            if len(self.trainer.loggers) == 2:
                self.trainer.loggers[1].log_metrics(
                    {
                        "grad/min": grad_norms.min().item(),
                        "grad/max": grad_norms.max().item(),
                        "grad/mean": grad_norms.mean().item(),
                        "grad/median": grad_norms.median().item(),
                    },
                    step=self.trainer.global_step,
                )

    def get_experiment_name(self, date_str: str, extra_name: str):
        EXPERIMENTS_PATH = "/Experiments/experiments/"
        return f"{EXPERIMENTS_PATH}/{date_str}_{self.hparams['datamodule']['name']}_{self.hparams['model']['name']}{extra_name}"

    def train_dataloader(self):
        dl = self.dm.train_dataloader()
        self.current_train_dl = dl
        return dl

    def val_dataloader(self):
        dl = self.dm.val_dataloader()
        self.current_val_dl = dl
        return dl

    def test_dataloader(self):
        dl = self.dm.test_dataloader()
        self.current_test_dl = dl
        return dl

    def copy_best_artifacts(self, metric, best):
        # if pq.cpu().item() > self.best_pq:
        if metric < best or self.best == -1:
            self.best = metric
            # source_dir = os.path.join(
            #     self.logger.save_dir,
            #     self.logger.experiment_id,
            #     self.logger.run_id,
            #     "artifacts",
            #     "current",
            # )
            # destination_dir = os.path.join(
            #     self.logger.save_dir,
            #     self.logger.experiment_id,
            #     self.logger.run_id,
            #     "artifacts",
            #     f"best_{self.trainer.checkpoint_callback.monitor}",
            # )
            source_dir = os.path.join(self.log_folder, "current")
            destination_dir = os.path.join(
                self.log_folder, f"best_{self.trainer.checkpoint_callback.monitor}"
            )
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            for item in os.listdir(source_dir):
                shutil.copy2(
                    os.path.join(source_dir, item), os.path.join(destination_dir, item)
                )

    def fit_estimator(self, dataloader, **kwargs):
        assert hasattr(
            self.model, "autoencoder"
        ), "this function is only for autoencoder based models"

        zs = []

        with torch.no_grad():  # Disable gradient calculations
            for batch in tqdm(dataloader):
                x = self.dm.get_image_from_batch(batch)
                x = x.cuda()  # Move to GPU
                latent = self.get_latent(x)
                zs.append(latent["z"].cpu())  # Move result to CPU and store

        z = torch.concat(zs)

        self.estimator.fit(z, **kwargs)

    @torch.no_grad()
    def predict_confidence(self, x, beta=-3.5, device="cuda"):
        self.eval()
        if isinstance(x, dict) and "z" in x:
            z = x["z"]
            out = x
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x.numpy()).to(torch.float32)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            if x.shape[1] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            x = x.to(device)
            h, w = x.shape[-2], x.shape[-1]

            # Resize if height and width are not 256 or 512
            if [w, h] != self.dm.base_size:

                x = F.interpolate(
                    x,
                    size=self.dm.base_size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
            out = self(x)
            z = out["z"]
        density = self.estimator.predict(z).cpu()
        uncertainty = torch.stack(
            [
                self.model.decoder.semantic_head.get_uncertainty(out, i)
                for i in range(out["semantic"].shape[0])
            ]
        ).cpu()
        return beta * density - (1 - beta) * uncertainty.cpu().detach().mean(-1).mean(
            -1
        )
