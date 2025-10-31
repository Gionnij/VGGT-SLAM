import os
import torch
from tqdm import tqdm
from torchvision import transforms

torch.set_grad_enabled(False)

from Prior2Former.lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
import argparse

torch.set_grad_enabled(False)
from Benchmarks.Open_World_Benchmark.segmentmeifyoucan.road_anomaly_benchmark.evaluation import Evaluation
from Benchmarks.Open_World_Benchmark.estimators import get_estimator, dropoutUncertainty, Uncertainty


parser = argparse.ArgumentParser(description="Argument parser for start_smiyc_eval.py")

# Adding arguments
parser.add_argument("-d", "--dataset", required=True, help="Name of the dataset")
parser.add_argument("-n", "--name", required=True, help="Name of the estimator")
parser.add_argument("--ckpt", required=True, help="Path to the checkpoint file")
# load a list of args by parsing with json loads
parser.add_argument(
    "--use_stored_forward",
    "-f",
    default=False,
    type=bool,
    help="Use stored forward path",
)

# Parsing arguments
args = parser.parse_args()

modul = PanopticSegmentationModule.load_from_checkpoint(args.ckpt)
print(modul)
modul = modul.to("cuda")
modul.eval()

args.dataset = os.path.expanduser(args.dataset)
args.name = args.name.split(",")


evaluators = [
    Evaluation(name, dataset_name=args.dataset, threaded_saver=False)
    for name in args.name
]

out_size = next(iter(evaluators[0].get_frames())).image.shape[-3:-1]
print(args.dataset, out_size)
estimators = []
for name in args.name:
    names = name.split("-")
    if len(names) == 1:
        estimator = get_estimator(names[0], {"out_size": out_size})
    else:
        for arg in names[1:]:
            assert "=" in arg and len(arg.split("=")) == 2, f"Invalid argument {arg}"
        kwargs = {arg.split("=")[0]: float(arg.split("=")[1]) for arg in names[1:]}
        kwargs["out_size"] = out_size
        estimator = get_estimator(names[0], kwargs)
    estimators.append(estimator)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # v2.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=modul.dm.mean, std=modul.dm.std),
    ]
)


# Define function to process each frame
def process_frame(out, im, estimator, evaluator):
    if isinstance(estimator, dropoutUncertainty) or isinstance(estimator, Uncertainty):
        result = estimator(modul.model, x=im).cpu().numpy()
    else:
        result = estimator(modul.model, out=out).cpu().numpy()
    evaluator.save_output(frame, result)


for frame in tqdm(evaluators[0].get_frames()):
    if args.use_stored_forward:
        print("used stored")
        out = evaluators[0].read_forward(frame)
    else:
        im = transform(frame.image).cuda().unsqueeze(0)
        out = modul(im)

    for evaluator, estimator in zip(evaluators, estimators):
        process_frame(out, im, estimator, evaluator)
    del out, im
