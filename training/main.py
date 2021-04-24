import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from video_classification.dataset.image_dataset import ImageFolderDatasetPreload

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epoch", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")


def train(manager, args, model, device, train_loader):
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration(step_optimizers=["main"]):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.nll_loss(output, target)
                ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()


def test(args, model, device, data, target):
    """The extension loops over the iterator in order to
    drive the evaluator progress bar and reporting
    averages
    """
    model.eval()
    test_loss = 0
    correct = 0
    data, target = data.to(device), target.to(device)
    output = model(data)
    # Final result will be average of averages of the same size
    test_loss += F.nll_loss(output, target, reduction="mean").item()
    ppe.reporting.report({"val/loss": test_loss})
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    ppe.reporting.report({"val/acc": correct / len(data)})


def main():
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mean = np.loadtxt("../data/mean.txt")
    sigma = np.loadtxt("../data/sigma.txt")
    normalize = transforms.Normalize(mean=mean, std=sigma)

    transform_fn = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = ImageFolderDatasetPreload(args.data, transform_fn)

    train_indices, val_indices = train_test_split(
        list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
