# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""



import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader 

from typing import Iterable, Optional
from time import time
from pkbar import Kbar

from core.criterion.Metric import calc_iou

def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    loss_fn: nn.Module,
    epoch: int,
    args: object,
    log: object,
    device: object
):
    model.train()

    progress = Kbar(
        target=len(data_loader)
    )
    
    if args.gpu == 0:
        print(f"\n[INFO] current epoch is : {epoch}")

    for batch_idx, data in enumerate(data_loader):

        img, label = data
        img = img.to(
            device=device,
            non_blocking=True
            )
        label = label.to(
            device=device,
            non_blocking=True
        )
   
        # forward the function and extract the pred labels with automatic mixed-precision
        with torch.cuda.amp.autocast():
            pred_label = model(img)

            # calculation the loss and backpropagate for update weights.
            loss = loss_fn(
                pred_label,
                label
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value): 
            print(
                f"[Error] current loss is {loss_value}, stopping training to avoid overflow attacks"
                )
            sys.exit(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()

        logger(
            batch_idx=batch_idx,
            epoch=epoch,
            gpu_rank=args.rank,
            loss=loss_value,
            model=model,
            progress_bar=progress,
            args=args
        )

    if scheduler is not None:
        scheduler.step()

    progress.add(1)


def evaluate(
    model: nn.Module,
    data_loader: Iterable,
    loss_fn: nn.Module,
    epoch: int,
    device:object,
    args:object
):
    model.eval()

    for batch_idx, data in enumerate(data_loader):
        
        all_loss = float(0)
        all_iou = float(0)

        img, label = data
        img = img.to(
            device=device,
            non_blocking=True
            )
        label = label.to(
            device=device,
            non_blocking=True
        )

        # forward the function and extract the pred labels with automatic mixed-precision
        with torch.cuda.amp.autocast():
            pred_label = model(img)

            # calculation the loss and backpropagate for update weights.
            loss = loss_fn(
                pred_label,
                label
            )

        iou_value = calc_iou(
            ground_thruth=label,
            prediction=pred_label
        )


        all_loss += loss.item()
        all_iou += iou_value

    if args.rank == 0: 
        print(f'[INFO] Validation Loss is : {all_loss/len(data_loader)}  IoU Score is : {all_iou/len(data_loader)}')

        #TODO: have to build metrics like, IoU 
    torch.cuda.synchronize()

def logger(
    batch_idx:int,
    epoch:int,
    gpu_rank: int,
    loss: float,
    model: nn.Module,
    progress_bar: object,
    args: object
):
    if gpu_rank == 0:
        progress_bar.update(
            batch_idx,
            values=[("loss: ", loss)] 
        )

    if epoch % 100 == 0:
        torch.save(
            model.state_dict(), 
            args.save_model_path+f'/{epoch}_v4_ce.pt'
        )
