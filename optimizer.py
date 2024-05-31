# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
import torch.nn as nn
from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = []
    skip_keywords = []
    if config.no_decay is not None:
        skip_keywords = skip_keywords + config.no_decay.split(',')
    parameters = set_weight_decay(model, skip, skip_keywords)

    optimizer = None
    if config.optim == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.momentum,
                              lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optim == 'adamw':
        b1, b2 = map(float, config.betas.split(','))
        optimizer = optim.AdamW(parameters, betas=(b1, b2),
                                lr=config.learning_rate, weight_decay=config.weight_decay)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name in skip_list) or check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    if len(no_decay) > 0:
        print('**** some para wo decay ****')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def lr_warmup(base_lr, iter, warmup_iter=1500, warmup_ratio=1e-6):
    if iter >= warmup_iter:
        return base_lr
    else:
        return base_lr * (1- (1 - float(iter) / warmup_iter) * (1 - warmup_ratio))
     

def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power, warmup):
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    if warmup>0:
        lr = lr_warmup(lr, i_iter, warmup_iter=warmup)
    # lr = max(lr, 1e-4)
    # optimizer.param_groups[0]['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
