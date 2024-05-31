import torch.nn as nn
import math
import torch
import numpy as np
from torch.nn import functional as F

from .ohem import CriterionOhemDSN
import torch.distributed as dist


def build_criterions(loss_type, dataset, loss_para):
    if len(loss_type.split(','))>1:
        return CombinedCriterion(loss_type, dataset, loss_para)
    else:
        return build_criterion(loss_type, dataset, loss_para)

def build_criterion(loss_type, dataset, loss_para):
    if loss_type=='ce':
        Criterion=CriterionDSN
    elif loss_type=='ohem':
        Criterion=CriterionOhemDSN
    elif loss_type=='gsrl':
        Criterion=CriterionGsrlDSN
    else:
        raise NotImplementedError(loss_type)
        
    return Criterion(dataset=dataset, **loss_para)

        
class CombinedCriterion(nn.Module):
    def __init__(self, loss_types, dataset=None, loss_para={}):
        super(CombinedCriterion, self).__init__()
        self.criterions = []
        for loss_type in loss_types.split(','):
            self.criterions.append(build_criterion(loss_type, dataset, loss_para))
        
    def forward(self, preds, labels):
        loss = {}
        loss_ = 0.0
        for criterion in self.criterions:
            loss_tmp = criterion(preds, labels)
            loss_ = loss_ + loss_tmp['loss']
            loss.update(loss_tmp)
        loss['loss'] = loss_
        return loss
        
        
class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, dataset=None, ds_weight=0.4, balance_weight=False, **kwargs):
        super(CriterionDSN, self).__init__()
        self.ignore_index = dataset.ignore_label
        self.ds_weight = ds_weight
        if balance_weight:
            weight = dataset.class_weights
        else:
            weight = None
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction='mean')

    def forward(self, preds, target):
        if isinstance(preds,list):
            if len(preds) >= 2:
                loss = self.criterion(preds[0], target)
                loss_ds = self.criterion(preds[1], target)
                loss = loss + loss_ds*self.ds_weight
            else:
                loss = self.criterion(preds[0], target)
        elif isinstance(preds,dict):
            loss = self.criterion(preds['pred'], target['ori'])
            loss_ds = self.criterion(preds['deepsup'], target['ori'])
            loss = loss + loss_ds*self.ds_weight
        return {'loss':loss}


class CriterionGsrlDSN(nn.Module):
    def __init__(self, dataset=None, ds_weight=0.4, k=9, gamma=9, **kwargs):
        super(CriterionGsrlDSN, self).__init__()
        self.k = k
        self.gamma = gamma
        self.ignore_index = dataset.ignore_label
        self.ds_weight = ds_weight
        self.balance_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, preds, labels):
        with torch.no_grad():
            weight = F.max_pool2d(labels['weight'].unsqueeze(1), self.k, stride=1, padding=self.k//2)[:, 0]
            score = torch.softmax(preds[0], 1)
            sort_score = torch.sort(score, dim=1, descending=True)[0]
            calibrate_factor = 1 + self.gamma * (1 - (sort_score[:, 0] - sort_score[:, 1]))
            weight = calibrate_factor * weight
            weight[labels['ori'] == self.ignore_index] = 0.0
        loss1 = self.balance_criterion(preds[0], labels['ori'])
        loss1 = (loss1 * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + 1e-8)
        loss1 = torch.mean(loss1)
        loss_ds = self.balance_criterion(preds[1], labels['ori'])
        loss_ds = (loss_ds * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + 1e-8)
        loss_ds = torch.mean(loss_ds)
        loss = loss1 + self.ds_weight * loss_ds
        return {'loss':loss}
