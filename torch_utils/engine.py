import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from utils.general import save_validation_results
import numpy as np
from tqdm.auto import tqdm


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    train_loss_hist,
    print_freq,
    scaler=None,
    scheduler=None,
    lr_scheduler=None,
    lr=0.001,
    epochs=5,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    batch_classification_list = []
    batch_bbox_reg_list = []

    if lr_scheduler == "one-cycle":
        print("scheduler one-cycle")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=epochs,
            max_lr=lr,
            steps_per_epoch=len(data_loader),
        )
    elif lr_scheduler == "cosine-annealing":
        print("scheduler cosine-annealing")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif lr_scheduler is None:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step_counter += 1
        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets
        ]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)

        # List of Faster R-CNN loss keys and corresponding lists
        faster_rcnn_losses = {
            "loss_classifier": batch_loss_cls_list,
            "loss_box_reg": batch_loss_box_reg_list,
            "loss_objectness": batch_loss_objectness_list,
            "loss_rpn_box_reg": batch_loss_rpn_list,
        }

        # List of RetinaNet loss keys and corresponding lists
        retinanet_losses = {
            "classification": batch_classification_list,
            "bbox_regression": batch_bbox_reg_list,
        }

        # Handle Faster R-CNN losses
        for loss_name, target_list in faster_rcnn_losses.items():
            if (
                loss_name in loss_dict_reduced
                and loss_dict_reduced[loss_name] is not None
            ):
                target_list.append(loss_dict_reduced[loss_name].detach().cpu())

        # Handle RetinaNet losses
        for loss_name, target_list in retinanet_losses.items():
            if (
                loss_name in loss_dict_reduced
                and loss_dict_reduced[loss_name] is not None
            ):
                target_list.append(loss_dict_reduced[loss_name].detach().cpu())

        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (step_counter / len(data_loader)))

    return (
        metric_logger,
        batch_loss_list,
        batch_loss_cls_list,
        batch_loss_box_reg_list,
        batch_loss_objectness_list,
        batch_loss_rpn_list,
        batch_classification_list,
        batch_bbox_reg_list,
    )


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    device,
    save_valid_preds=False,
    out_dir=None,
    classes=None,
    colors=None,
):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if save_valid_preds and counter == 1:
            # The validation prediction image which is saved to disk
            # is returned here which is again returned at the end of the
            # function for WandB logging.
            val_saved_image = save_validation_results(
                images, outputs, counter, out_dir, classes, colors
            )
        elif save_valid_preds == False and counter == 1:
            val_saved_image = np.ones((1, 64, 64, 3))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats, val_saved_image
