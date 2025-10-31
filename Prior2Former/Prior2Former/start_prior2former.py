# Python
# Custom
import multiprocessing
import re
import traceback
import sys
import time

from lightning_utils.callbacks import (
    GradientNormMonitor,
    get_callback,
)
from lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
from utils.logging_utils.log_writers import (
    init_global_logger,
    deinit_global_logger,
    gl_error,
    gl_info,
)


from Config.parser_io import (
    parse_open_seg_config,
)



from pprint import pprint
import os
from datetime import datetime
import subprocess
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger


CUDA_VISIBLE_DEVICES = 0
DEBUG = os.environ.get("DEBUG", False)
ALL_EXPERIMENTS_PATH = os.environ.get(
    "EXPERIMENTS", os.path.join(os.path.expanduser("~"), "Experiments")
)
USE_MLFLOW = True
torch.set_float32_matmul_precision("medium")


def prepare_args(args):
    if "small_validation_size" in args["module"]:
        # for DDP strategy the validation size should be divided by the number of gpus
        args["module"]["small_validation_size"] /= args["trainer"]["gpus"]
    if args["trainer"]["gpus"] > 1:
        base_bs = 16
        if args["module"]["model"]["name"] == "mask2former":
            args["module"]["optimizer"]["args"]["lr"] = (
                args["module"]["optimizer"]["args"]["lr"]
                * args["trainer"]["gpus"]
                * args["datamodule"]["args"]["batch_size"]
                / base_bs
            )
        print("changed lr to ", args["module"]["optimizer"]["args"]["lr"])


    #check for correctness of arguments

    #checks that lr_scheduler has max_iters >= max_epochs, i.e. the scheduler assumes a training at least as long as the actual number of epochs
    if (
        hasattr(args, "module") and
        hasattr(args.module, "optimizer") and
        hasattr(args.module.optimizer, "lr_scheduler") and
        hasattr(args.module.optimizer.lr_scheduler, "args") and
        hasattr(args.module.optimizer.lr_scheduler.args, "max_iters") and
        hasattr(args, "trainer") and
        hasattr(args.trainer, "max_epochs")
    ):
        assert args.module.optimizer.lr_scheduler.args.max_iters >= args.trainer.max_epochs, (
            f"Expected max_iters ({args.module.optimizer.lr_scheduler.args.max_iters}) "
            f">= max_epochs ({args.trainer.max_epochs})"
        )



##
# Main
def log_git_hash(loggerml):
    def read_git_hash():
        try:
            print("read in git hash")
            githash = (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    check=True,
                    stdout=subprocess.PIPE,
                )
                .stdout.strip()
                .decode("utf-8")
            )
            print("log hash")
            loggerml.log_hyperparams({"hash": githash})
        except Exception as e:
            print(f"Error reading git hash: {e}")

    # Set the timeout duration in seconds
    timeout_duration = 1

    # Create a process to run the function
    process = multiprocessing.Process(target=read_git_hash)

    # Start the process
    process.start()

    # Wait for the process to complete with a timeout
    process.join(timeout=timeout_duration)

    # Check if the process is still alive (i.e., it timed out)
    if process.is_alive():
        print("The operation timed out")
        process.terminate()
        process.join()


def get_module(config):
    if "ood_datamodule" in config["module"]:
        config["module"].pop("ood_datamodule")
    if "type" in config:
        if config["type"] == "panoptic":
            return PanopticSegmentationModule(
                **config["module"], datamodule=config["datamodule"]
            )
        else:
            raise NotImplementedError("")


def get_experiment_name(_config, module):
    date_str = datetime.now()
    extra_name = ""
    if "extra_name" in _config:
        extra_name = f"_{_config['extra_name']}"
    experiment_name = module.get_experiment_name(date_str, extra_name)
    return experiment_name


def get_version(experiment_name):
    nums = []
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
        return os.path.join(experiment_name, "version_0")
    for d in os.listdir(experiment_name):
        if d.startswith("version"):
            nums.append(int(d.split("_")[-1]))
    if len(nums) == 0:
        return os.path.join(experiment_name, "version_0")
    else:
        return os.path.join(experiment_name, "version_" + str(1 + max(nums)))


def op_train(config, experiment_name=None, log=None, callbacks=[], local=False):
    pl.seed_everything(0)
    module = get_module(config)
    if experiment_name is None:
        experiment_name = get_experiment_name(config, module)

    # logger = TensorBoardLogger(save_dir=".", name=experiment_name, flush_secs=2)
    if log == "mlflow":
        assert False, "not up to date, check double logging"
        base = "/Experiments/mlflow/"
        # id = config_file.split("/")[-1].split(".")[0]
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            # save_dir=logdir,
            tracking_uri="file:" + base,
        )
        logger.log_hyperparams(
            {
                "hash": subprocess.check_output(["git", "rev-parse", "HEAD"])
                .strip()
                .decode("utf-8")
            }
        )
    elif log == "tensorboard":
        assert False, "not up to date, check double logging"
        logger = TensorBoardLogger(
            save_dir="/Experiments/tensorboard",
            name=experiment_name,
            flush_secs=2,
        )
        os.makedirs(logger.log_dir)
        with open(logger.log_dir + "/hash.txt", "w") as f:
            f.write(
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .strip()
                .decode("utf-8")
            )
    else:
        print("Initializing MLFLow logger")
        loggerml = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=config["run_name"],
            # save_dir=logdir,
            tracking_uri=config["trainer"]["tracking_uri"],
        )

        log_git_hash(loggerml)

        print("Initializing Tensorboard logger")
        loggert = TensorBoardLogger(
            save_dir=os.path.join(ALL_EXPERIMENTS_PATH, "tensorboard"),
            name=experiment_name,
            flush_secs=2,
        )
        logger = [loggerml, loggert]
    if module.has_lr_scheduler():
        callbacks.append(LearningRateMonitor("epoch"))
    if "checkpoint" in config["trainer"]:
        path = os.path.join(
            ALL_EXPERIMENTS_PATH,
            config["module"]["log_folder"],
            "current_ckpts",
        )
        os.makedirs(path, exist_ok=True)

        # Open the file within the specified directory
        with open(os.path.join(path, "hparams.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            if "monitor" not in config["trainer"]["checkpoint"]:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=path,
                        filename="{epoch}",
                        **config["trainer"]["checkpoint"],
                    )
                )
            elif "mean_val_loss" == config["trainer"]["checkpoint"]["monitor"]:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=path,
                        filename="{epoch}-{mean_val_loss:.2f}",
                        **config["trainer"]["checkpoint"],
                    )
                )
            elif "validation_pq" == config["trainer"]["checkpoint"]["monitor"]:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=path,
                        filename="{epoch}-{validation_pq:.2f}",
                        **config["trainer"]["checkpoint"],
                    )
                )
            elif "val_iou" == config["trainer"]["checkpoint"]["monitor"]:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=path,
                        filename="{epoch}-{val_iou:.2f}",
                        **config["trainer"]["checkpoint"],
                    )
                )
            else:
                raise NotImplementedError(
                    f"{config['trainer']['checkpoint']['monitor']} monitor metric unknown"
                )
    for callback in config["trainer"].get("callbacks", []):
        callbacks.append(get_callback(callback["name"], callback["args"]))
    callbacks.append(GradientNormMonitor())

    if "profiler" in config["trainer"]:

        if config["trainer"]["profiler"] == "simple":
            from pytorch_lightning.profilers import SimpleProfiler

            profiler = SimpleProfiler(
                filename="profiler",
                dirpath=os.path.join(
                    ALL_EXPERIMENTS_PATH, config["module"]["log_folder"]
                ),
            )
        else:
            profiler = None
    else:
        profiler = None
    print(module.model)
    trainer = pl.Trainer(
        devices=config["trainer"].get("gpus", 0),
        logger=logger,
        callbacks=callbacks,
        max_epochs=config["trainer"]["max_epochs"],
        log_every_n_steps=1,
        val_check_interval=(
            config["trainer"]["val_check_interval"]
            if "val_check_interval" in config["trainer"]
            else 1.0
        ),
        check_val_every_n_epoch=(
            config["trainer"]["val_interval"]
            if "val_interval" in config["trainer"]
            else 1
        ),
        num_sanity_val_steps=(
            config["trainer"]["num_sanity_val_steps"]
            if "num_sanity_val_steps" in config["trainer"]
            else 2
        ),
        profiler=profiler,
        **(config["trainer"].get("trainer_args", {})),
    )
    print("start fit")
    return (
        trainer.fit(
            module,
            ckpt_path=(
                config["trainer"]["ckpt_path"]
                if "ckpt_path" in config["trainer"]
                else None
            ),
        ),
        module,
    )


def extract_substring(arg):
    match = re.search(r"\.([^\.]+)=(.+)", arg)
    if match:
        return match.group(1) + "=" + match.group(2)
    return ""


def get_run_name():
    l = sys.argv
    s = l[2].split("/")[-1].split(".")[0]
    for p in l[3:]:
        if (
            p.startswith("--experiment")
            or not p.startswith("--")
            or p.endswith("}")
            or p.startswith("--trainer.ckpt_path")
            or p.startswith("--trainer.tracking")
            or p.startswith("--datamodule.args.data_dir")
        ):
            continue
        s += "_" + extract_substring(p)

    run_name = s + "_" + str(time.time())
    return run_name


if __name__ == "__main__":
    run_name = get_run_name()

    args = parse_open_seg_config()
    prepare_args(args)
    experiment_name = args["experiment_name"]
    experiment_path = os.path.join(experiment_name, run_name)
    args["run_name"] = run_name
    args["module"]["log_folder"] = experiment_path
    os.makedirs(os.path.join(ALL_EXPERIMENTS_PATH, experiment_path), exist_ok=True)
    if DEBUG:
        pprint(args)
        op_train(args, experiment_name=experiment_name)
    else:
        init_global_logger(
            root_path=ALL_EXPERIMENTS_PATH,
            experiment_path=experiment_path,
            use_fh=True,
        )
        gl_info(args)

        try:
            op_train(args, experiment_name=experiment_name)
        except Exception as e:
            experiment_path = os.path.join(ALL_EXPERIMENTS_PATH, experiment_path)
            os.rename(experiment_path, f"{experiment_path}-failed")

            gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")

            experiment_path = f"{experiment_path}-failed"
            raise e
        finally:
            deinit_global_logger()

