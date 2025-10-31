import warnings

from torch.optim.lr_scheduler import (
    _LRScheduler,
    MultiStepLR,
    CosineAnnealingLR,
    StepLR,
    # PolynomialLR,
)


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    # last_epoch is step not epoch - according to implementation
    def get_lr(self):
        return [
            max(
                base_lr * (1 - self.last_epoch / self.max_iters) ** self.power,
                self.min_lr,
            )
            for base_lr in self.base_lrs
        ]


def get_scheduler(trainings_config, optimizer):
    scheduler = trainings_config["scheduler"]["type"]
    scheduler_map = {
        "none": lambda opt: None,
        # "polynomial": lambda opt: PolynomialLR(
        #     opt,
        #     total_iters=trainings_config["scheduler"]["total_iters"],
        #     power=trainings_config["power"],
        # ),
        "poly": lambda opt: PolyLR(
            opt, max_iters=trainings_config["scheduler"]["max_iter"]
        ),
        "StepLR": lambda opt: StepLR(
            opt,
            step_size=trainings_config["scheduler"]["step_size"],
            gamma=trainings_config["scheduler"]["gamma"],
        ),
        "MultiStep": lambda opt: MultiStepLR(
            opt,
            milestones=trainings_config["milestones"],
            gamma=trainings_config["scheduler"].get("gamma", 0.2),
        ),
        "cosine": lambda opt: CosineAnnealingLR(opt, T_max=200),
    }
    return scheduler_map[scheduler](optimizer)


class SeqLR(_LRScheduler):
    """
    From Stefano
    """

    def __init__(
        self,
        optimizer,
        step_size,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
        initial_epochs=10,
        backbone_lr_factor=0.1,
    ):
        self.step_size = step_size
        self.gamma = gamma
        self.initial_epochs = initial_epochs
        super(SeqLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
