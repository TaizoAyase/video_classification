import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets
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
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
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
parser.add_argument("--lr-step", default=30, type=int, help="lr scheduler step (epoch)")
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
    "--preload", dest="preload", action="store_true", help="pre-load dataset images"
)
parser.add_argument(
    "--debug", dest="debug", action="store_true", help="debug with small dataset"
)
parser.add_argument("--test-frac", default=0.2, type=float, help="test data fraction")
parser.add_argument("--seed", default=777, type=int, help="seed for splitting.")
parser.add_argument("--gpu", default=-1, type=int, help="GPU id to use.")
parser.add_argument("--out", default="./results", type=str, help="Output dirname.")
parser.add_argument("--snapshot", default=None, type=str, help="Snapshot to resume")


def train(manager, args, model, lossfn, device, train_loader):
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            with manager.run_iteration(step_optimizers=["main"]):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = lossfn(output, target)
                ppe.reporting.report({"train/loss": loss.item()})
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                ppe.reporting.report({"train/acc": correct / len(data)})
                loss.backward()


def test(args, model, lossfn, device, data, target):
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
    test_loss += lossfn(output, target).item()
    ppe.reporting.report({"val/loss": test_loss})
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    ppe.reporting.report({"val/acc": correct / len(data)})


def main():
    args = parser.parse_args()

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"using device {device}")

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
    if args.preload:
        dataset = ImageFolderDatasetPreload(args.data, transform_fn)
    else:
        dataset = datasets.ImageFolder(args.data, transform_fn)

    train_indices, val_indices = train_test_split(
        list(range(len(dataset.targets))),
        test_size=args.test_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=dataset.targets,
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    if args.debug:
        n_train = 100
        n_test = 10
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [n_train, len(train_dataset) - n_train]
        )
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset, [n_test, len(val_dataset) - n_test]
        )

    kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    model = models.__dict__[args.arch]()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    lossfn = nn.CrossEntropyLoss().to(device)

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(update_interval=1),
        extensions.observe_lr(optimizer=optimizer),
        extensions.LRScheduler(scheduler, trigger=(args.lr_step, "epoch")),
        # extensions.ParameterStatistics(model, prefix="model"),
        # extensions.VariableStatisticsPlot(model),
        extensions.Evaluator(
            val_data_loader,
            model,
            eval_func=lambda data, target: test(
                args, model, lossfn, device, data, target
            ),
            progress_bar=True,
        ),
        extensions.PlotReport(["train/loss", "val/loss"], "epoch", filename="loss.png"),
        extensions.PlotReport(
            ["train/acc", "val/acc"], "epoch", filename="accuracy.png"
        ),
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "train/loss",
                "train/acc",
                "val/loss",
                "val/acc",
                "lr",
            ]
        ),
        extensions.snapshot(),
    ]
    trigger = None
    manager = ppe.training.ExtensionsManager(
        model,
        optimizer,
        args.epochs,
        out_dir=str(outdir),
        extensions=my_extensions,
        iters_per_epoch=len(train_data_loader),
        stop_trigger=trigger,
    )
    # Lets load the snapshot
    if args.snapshot is not None:
        state = torch.load(args.snapshot)
        manager.load_state_dict(state)
    with open(outdir / "args.json", "w") as f:
        json.dump(vars(args), f)
    train(manager, args, model, lossfn, device, train_data_loader)
    torch.save(model.state_dict(), str(outdir / "model.pt"))


if __name__ == "__main__":
    main()
