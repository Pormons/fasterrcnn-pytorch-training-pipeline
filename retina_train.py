from config import (
    DEVICE,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES,
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR,
)
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot, save_mAP
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset,
    create_valid_dataset,
    create_train_loader,
    create_valid_loader,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import yaml
import numpy as np
import torchinfo
import os
import torch
import matplotlib.pyplot as plt
import time
import os
from torch_utils import utils

torch.multiprocessing.set_sharing_strategy("file_system")

plt.style.use("ggplot")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="fasterrcnn_resnet50_fpn_v2", help="name of the model"
    )
    parser.add_argument("--data", default=None, help="path to the data config file")
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        help="computation/training device, default is GPU if GPU present",
    )
    parser.add_argument(
        "-e", "--epochs", default=5, type=int, help="number of epochs to train for"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        help="number of workers for data processing/transforms/augmentations",
    )
    parser.add_argument(
        "-b", "--batch", default=4, type=int, help="batch size to load the data"
    )
    parser.add_argument(
        "--lr", default=0.001, help="learning rate for the optimizer", type=float
    )
    parser.add_argument(
        "-ims",
        "--imgsz",
        default=640,
        type=int,
        help="image size to feed to the network",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        type=str,
        help="training result dir name in outputs/training/, (default res_#)",
    )
    parser.add_argument(
        "-vt",
        "--vis-transformed",
        dest="vis_transformed",
        action="store_true",
        help="visualize transformed images fed to the network",
    )
    parser.add_argument(
        "--mosaic",
        default=0.0,
        type=float,
        help="probability of applying mosaic, (default, always apply)",
    )
    parser.add_argument(
        "-uta",
        "--use-train-aug",
        dest="use_train_aug",
        action="store_true",
        help="whether to use train augmentation, blur, gray, \
              brightness contrast, color jitter, random gamma \
              all at once",
    )
    parser.add_argument(
        "-ca",
        "--cosine-annealing",
        dest="cosine_annealing",
        action="store_true",
        help="use cosine annealing warm restarts",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="path to model weights if using pretrained weights",
    )
    parser.add_argument(
        "-r",
        "--resume-training",
        dest="resume_training",
        action="store_true",
        help="whether to resume training, if true, \
            loads previous training plots and epochs \
            and also loads the otpimizer state dictionary",
    )
    parser.add_argument(
        "-st",
        "--square-training",
        dest="square_training",
        action="store_true",
        help="Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.",
    )
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up the distributed training",
    )
    parser.add_argument(
        "-dw",
        "--disable-wandb",
        dest="disable_wandb",
        action="store_true",
        help="whether to use the wandb",
    )
    parser.add_argument(
        "--sync-bn", dest="sync_bn", help="use sync batch norm", action="store_true"
    )
    parser.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )
    parser.add_argument(
        "--patience",
        default=10,
        help="number of epochs to wait for when mAP does not increase to \
              trigger early stopping",
        type=int,
    )
    parser.add_argument("--seed", default=0, type=int, help="golabl seed for training")
    parser.add_argument(
        "--project-dir",
        dest="project_dir",
        default=None,
        help="save resutls to custom dir instead of `outputs` directory, \
              --project-dir will be named if not already present",
        type=str,
    )

    args = vars(parser.parse_args())
    return args


# Function for running training iterations.
def train(train_data_loader, model, scaler=None):
    print("Training")
    model.train()

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced.item()

        train_loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value


# Function for running validation iterations.
def validate(valid_data_loader, model):
    print("Validating")
    model.eval()

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict["boxes"] = targets[i]["boxes"].detach().cpu()
            true_dict["labels"] = targets[i]["labels"].detach().cpu()
            preds_dict["boxes"] = outputs[i]["boxes"].detach().cpu()
            preds_dict["scores"] = outputs[i]["scores"].detach().cpu()
            preds_dict["labels"] = outputs[i]["labels"].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    train_dataset = create_train_dataset(TRAIN_DIR)
    valid_dataset = create_valid_dataset(VALID_DIR)
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.1, verbose=True)

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Mame to save the trained model with.
    MODEL_NAME = "model"

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image

        show_tranformed_image(train_loader)

    # To save best model.
    save_best_model = SaveBestModel()

    metric = MeanAveragePrecision()

    # Training loop.
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch+1} mAP: {metric_summary['map']}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary["map_50"])
        map_list.append(metric_summary["map"])

        # save the best model till now.
        save_best_model(model, float(metric_summary["map"]), epoch, "outputs")
        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        # scheduler.step()
