import csv
import json
import os
import pickle
import random
from typing import Dict, Any

import mlflow
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.logging_utils.log_writers import gl_error, gl_info


def write_json(path, filename, content):
    file_path = os.path.join(path, filename)
    with open(file_path,"w") as f:
        json.dump(content,f)


class CsvWriter():
    def __init__(self, path, filename, content):
        self.path = path
        self.filename = filename
        write_csv_header(path, filename, content)


def append_csv_row(path, filename, content):
    file_path=os.path.join(path,filename+".csv")
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = content.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow(content)


def write_csv_header(path, filename, content):
    file_path=os.path.join(path,filename+".csv")
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = content.keys()
        writer = csv.writer(csvfile)

        writer.writerow(fieldnames)


def snake_case_to_minus_case(parameters: Dict[str, Any]):
    return {p.replace("_", "-"): v for p, v in parameters.items()}


def setup_random(seed=None):
    if seed == -1 or seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    print("Worker Seed")
    print(torch.initial_seed())
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrackingWriter():

    def log_metrics(self, key,value,cycle):
        pass

    def log_config(self,config):
        pass

    def log_text(self,text, name):
        pass

    def mark_failed(self):
        pass

    def log_figure(self, key, fig, cycle):
        pass

    def log_image(self,key,img,cycle):
        pass

    def log_histogram(self, key, hist, cycle):
        pass

    def log_file(self,filename):
        pass

    def write_file(self,content,filename):
        pass

global Writer
#Writer dataclass : SummaryWriter
Writer  = None#: Union[Dict[str,TrackingWriter],None] = None


def get_global_writer():
    global Writer
    if not Writer:
        raise LookupError("TensorboardWriterNotInitialized")


def global_write_scalar(key,value,cycle):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_metrics(key, value, cycle)
        except:
            gl_error(f"Error during writing metric to {writer.__name__}")


def global_write_config(config):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_config(config)
        except:
            gl_error(f"Error during writing config to {writer.__name__}")


def global_write_text(text, name):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_text(text, name)
        except:
            gl_error(f"Error during writing text to {writer.__name__}")


def global_writer_mark_failed():
    global Writer
    for writer in Writer.values():
        try:
            writer.mark_failed()
        except:
            gl_error(f"Error during mark failed to {writer.__name__}")


def global_write_fig(key,fig,cycle):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_figure(key,fig,cycle)
        except:
            gl_error(f"Error during writing figure to {writer.__name__}")


def global_write_images(key,img,cycle):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_image(key,img,cycle)
        except:
            gl_error(f"Error during writing image to {writer.__name__}")


def global_write_histogram(key,value,cycle):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_histogram(key,value,cycle)
        except:
            gl_error(f"Error during writing histogram to {writer.__name__}")


def global_log_file(filename):
    global Writer
    for writer in Writer.values():
        try:
            writer.log_file(filename)
        except:
            gl_error(f"Error during logging file to {writer.__name__}")

def global_write_file(content, filename):
    global Writer
    for writer in Writer.values():
        try:
            writer.write_file(content,filename)
        except:
            gl_error(f"Error during writing file to {writer.__name__}")



def check_mlflow_mounted():
    return os.path.exists("/ad-vantage/data/store/training/algoritmi/mlflow/tracking")


TF_BOARD = "tf_board"


def __init_global_writer():
    global Writer
    if Writer is None:
        Writer = {}

def init_tfboard_writer(root_path, experiment_path):
    __init_global_writer()
    global Writer
    Writer["tfboard"] =TensorBoardWriter(root_path,experiment_path)


def init_mlflow_writer(flow_client):
    __init_global_writer()
    global Writer
    Writer["mlflow"] = MLFLOWWriter(flow_client)


def init_file_writer(root_path, experiment_path):
    __init_global_writer()
    global Writer
    Writer["file"] = FileWriter(root_path,experiment_path)

def track_selected_images(selected_unlabeled_idx, train_dataset, cycle_num):
    for i, idx in enumerate(selected_unlabeled_idx):
        try:
            img = train_dataset.get_image_visu(idx)
        except AttributeError:
            img, _ = train_dataset[idx]
        global_write_images(f"Images/SelectedImage/C{cycle_num}-{i}", img, i)


class MLFLOWWriter(TrackingWriter):
    __name__="MLFLOWWriter"
    def __init__(self,flow_client:mlflow):
        print(f"{self.__name__} initialized")
        self.mlflow_handler=flow_client
    def log_metrics(self, key,value,cycle):
        self.mlflow_handler.log_metric(key,value,cycle)

    def log_config(self, config):

        self.mlflow_handler.log_params(config)

    def log_text(self, text, name):
        self.mlflow_handler.log_text(text, name+".txt")

    def mark_failed(self):
        self.mlflow_handler.set_tag("LOG_STATUS", "FAILED")

    def log_figure(self, key, fig, cycle):
        self.mlflow_handler.log_figure(fig, f"{key}-{cycle}.pdf")

    def log_image(self, key, img, cycle):
        self.mlflow_handler.log_image(np.moveaxis(img, [0], [2]), f"{key}-{cycle}.jpg")

    def log_file(self, filename):
        self.mlflow_handler.log_artifact(filename)

class TensorBoardWriter(TrackingWriter):
    __name__="TensorBoardWriter"
    def __init__(self,root_path,experiment_path):
        logging_folder = os.path.join(root_path, TF_BOARD, experiment_path)
        os.makedirs(logging_folder, exist_ok=True)
        gl_info(f"{self.__name__} initialized logging to : {logging_folder}")
        self.tf_handler =  SummaryWriter(log_dir=logging_folder, comment=experiment_path)

    def log_metrics(self, key,value,cycle):
        self.tf_handler.add_scalar(key, value, cycle)

    def log_config(self, config):
        self.tf_handler.add_hparams(config, {}, hparam_domain_discrete=None, run_name=None)

    def log_text(self, text, name):
        self.tf_handler.add_text(name, text)

    def log_figure(self, key, fig, cycle):
        self.tf_handler.add_figure(key,fig,cycle)

    def log_image(self, key, img, cycle):
        self.tf_handler.add_image(key,img,cycle)

    def log_histogram(self, key, hist, cycle):
        self.tf_handler.add_histogram(key,hist,cycle)


class FileWriter(TrackingWriter):
    __name__="FileWriter"
    def __init__(self,logging_folder, experiment_path):
        self.logging_folder :str = os.path.join(logging_folder,experiment_path)
        gl_info(f"{self.__name__} initialized logging to : {self.logging_folder}")
        os.makedirs(os.path.join(self.logging_folder, "Figures"), exist_ok=True)
        os.makedirs(os.path.join(self.logging_folder, "Images"), exist_ok=True)
        self.stream_file = os.path.join(self.logging_folder, "tracking_stream.txt")

    def log_metrics(self, key,value,cycle):
        with open(self.stream_file, 'a') as f:
            f.write(f'{key}-{cycle}::{value}\n')

    def log_config(self,config):
        pass

    def log_text(self,text, name):
        try:
            with open(self.stream_file, 'a') as f:
                f.write(f'{name}::{text}\n')
        except:
            gl_error(f"Error during writing text to trackfile {name}::{text}")
    def log_figure(self, key, fig, cycle=""):
        key = key.split("/")[-1]
        file_name=f"{key}-{cycle}"
        fig.savefig(os.path.join(self.logging_folder,"Figures",f"{file_name}.pdf"), bbox_inches='tight')
        with open(f'{self.logging_folder}/Figures/{file_name}.pickle','wb') as f:
            pickle.dump(fig, f)

    def log_image(self,name,img,cycle):
        name = name.split("/")[-1]
        file_name = f"{name}-{cycle}"
        from PIL import Image
        im = Image.fromarray(img)
        im.save(os.path.join(self.logging_folder,"Images",f"{file_name}.png"))

    def write_file(self,content,filename):
        from numpy import save as npsave
        if ".npy" in filename:
            npsave(filename,content)