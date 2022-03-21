# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
import argparse

from matplotlib.pyplot import axis
from typing import Dict, List, Union
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchcontrib.optim import SWA

def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    
    running_loss = 0
    num_total = 0.0
    train_acc, correct_train, target_count = 0, 0, 0
    ii = 0
    model.train()
    scaler = GradScaler()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_y = batch_y.view(-1).to(device)
        _, batch_out = model(batch_x)#, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        scaler.scale(batch_loss).backward()
        # batch_loss.backward()
        scaler.step(optim)
        scaler.update()
        # optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))
        
        # accuracy
        _, predicted = torch.max(batch_out.data, 1)
        target_count += batch_y.size(0)
        correct_train += (batch_y == predicted).sum().item()
        train_acc = (100 * correct_train) / target_count

    running_loss /= num_total
    return running_loss, train_acc


def eval_epoch(
    trn_loader: DataLoader,
    model,
    device: torch.device,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    val_acc, correct_train, target_count = 0, 0, 0
    ii = 0
    model.eval()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        # batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_y = batch_y.view(-1).to(device)
        _, batch_out = model(batch_x) #model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        
        # accuracy
        _, predicted = torch.max(batch_out.data, 1)
        target_count += batch_y.size(0)
        correct_train += (batch_y == predicted).sum().item()
        val_acc = (100 * correct_train) / target_count
        
    running_loss /= num_total
    return running_loss, val_acc



def main(args):

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    label_path = Path(config['label_path'])

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    # if device == "cpu":
    #     raise ValueError("GPU not detected!")

    # define model architecture
    model_config = args.config["model_config"]
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, eval_loader = get_loader(database_path, label_path, config)

    # get optimizer and scheduler
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # Training
    best_acc = 0.0
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        training_loss, training_acc = train_epoch(trn_loader, model, optimizer, device, scheduler, config)
        eval_loss, eval_acc = eval_epoch(eval_loader, model, device, config)
        print(f'[{epoch}] Training Loss : {training_loss} / Training Accuracy : {training_acc} | Eval Loss : {eval_loss} / Eval Accuracy : {eval_acc}')
        if eval_acc >= best_acc:
            best_model = model.state_dict()
    
    y, pred = [], []
    model.load_state_dict(best_model)
    for batch_x, batch_y in trn_loader:
        batch_x = batch_x.to(device)
        # batch_y = batch_y.view(-1).type(torch.int64).to(device)
        with torch.no_grad():
            batch_y = batch_y.view(-1).to(device)
            _, batch_out = model(batch_x)
        pred.extend(list(batch_out.cpu().numpy()))
        y.extend(list(y.cpu().numpy()))

    conf_metrics = confusion_matrix(pred, y)
    
    return best_model, conf_metrics