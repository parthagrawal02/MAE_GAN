# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
from adap_weight import aw_loss
from utils.misc import plot_reconstruction



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples,_) in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        if args.cuda is not None:
            with torch.cuda.amp.autocast():
                mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = model(samples.to(device), mask_ratio=args.mask_ratio)
        else:
            mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = model(samples, mask_ratio=args.mask_ratio)

        print(model.parameters())
        print(mae_loss)
        print(disc_loss)
        print(adv_loss)
        gen_loss = aw_loss(mae_loss, adv_loss, optimizer, model)
        print(gen_loss)
        gen_loss_value = gen_loss.item()
        if not math.isfinite(gen_loss_value):
            print("Loss is {}, stopping training".format(gen_loss_value))
            sys.exit(1)

        gen_loss = gen_loss/accum_iter

        loss_scaler(gen_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, retain_graph = True)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = model(samples, mask_ratio=args.mask_ratio)

        disc_loss_value = disc_loss.item()

        if not math.isfinite(disc_loss_value):
            print("Loss is {}, stopping training".format(disc_loss_value))
            sys.exit(1)

        disc_loss = disc_loss/accum_iter
        loss_scaler(disc_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, retain_graph = True)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if args.cuda is not None:
            torch.cuda.synchronize()

        metric_logger.update(disc_loss=disc_loss_value)
        metric_logger.update(gen_loss=gen_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        disc_loss_value_reduce = misc.all_reduce_mean(disc_loss_value)
        gen_loss_value_reduce = misc.all_reduce_mean(gen_loss_value)


        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('disc_train_loss', disc_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('gen_train_loss', gen_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_figure('Reconstructed vs. actuals',
                            plot_reconstruction(currupt_img, samples),
                            global_step=epoch * len(data_loader) + data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
