# Copyright 2023 by Ismail Khalfaoui-Hassani, ANITI Toulouse.
#
# All rights reserved.
#
# This file is part of the Dcls-Audio package, and
# is released under the "MIT License Agreement".
# Please see the LICENSE file that should have been included as part
# of this package.


import math
from math import sqrt
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score
from timm.data import Mixup
from timm.utils import ModelEma

import utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    wandb_logger=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    use_amp=False,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:  # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = None  # (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log(
                {
                    "Rank-0 Batch Wise/train_loss": loss_value,
                    "Rank-0 Batch Wise/train_max_lr": max_lr,
                    "Rank-0 Batch Wise/train_min_lr": min_lr,
                },
                commit=False,
            )
            if class_acc:
                wandb_logger._wandb.log(
                    {"Rank-0 Batch Wise/train_class_acc": class_acc}, commit=False
                )
            if use_amp:
                wandb_logger._wandb.log(
                    {"Rank-0 Batch Wise/train_grad_norm": grad_norm}, commit=False
                )
            wandb_logger._wandb.log({"Rank-0 Batch Wise/global_train_step": it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, criterion, model, device, use_amp=False):
    criterion = criterion

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    target_all, output_all = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for batch in metric_logger.log_every(data_loader, 10, header):
        waveform = batch[0]
        target = batch[-1]

        waveform = waveform.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(waveform)
                loss = criterion(output, target)
        else:
            output = model(waveform)
            loss = criterion(output, target)

        target, output = target.int(), output
        target_all = torch.cat((target_all, target.clone().to(device)), 0)
        output_all = torch.cat((output_all, output.clone().to(device)), 0)

        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    gather_target_list = [
        torch.zeros_like(target_all) for _ in range(utils.get_world_size())
    ]
    gather_output_list = [
        torch.zeros_like(output_all) for _ in range(utils.get_world_size())
    ]

    dist.all_gather(gather_target_list, target_all)
    dist.all_gather(gather_output_list, output_all)

    output = torch.cat(gather_output_list, 0).cpu()
    target = torch.cat(gather_target_list, 0).int().cpu()

    mAP = average_precision_score(target, output)
    AUC = roc_auc_score(target, output, average=None)
    d_prime = sqrt(2) * norm.ppf(AUC).mean()
    AUC = AUC.mean()

    metric_logger.update(mAP=mAP.item(), AUC=AUC.item(), d_prime=d_prime.item())

    print(
        "* mAP {mAP:.3f} AUC {AUC:.3f} d_prime {d_prime:.3f} loss {losses.global_avg:.3f}".format(
            mAP=mAP, AUC=AUC, d_prime=d_prime, losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
