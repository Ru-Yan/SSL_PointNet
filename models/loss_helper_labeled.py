""" Loss functions on labeled data

Author: Zhao Na, 2019
Modified by Yezhen Cong, 2020
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import loss



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def get_labeled_loss(end_points):
    supervised_mask = end_points['supervised_mask']
    supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()   #假设label为2 [0,1]

    gt_labels=end_points['labels'][supervised_inds, ...]         #2*20000
    gt_labels = gt_labels.long()

    pred_feature=end_points['pred_feature'][supervised_inds, ...]           #2*20000*2

    criterion = nn.CrossEntropyLoss()   #torch的预测方法是，输入那个点的特征n*c玩意


    label_loss = criterion(pred_feature.view(gt_labels.numel(), -1), gt_labels.view(-1))

    _, classes = torch.max(pred_feature, -1)

    acc = (classes == gt_labels).float().sum() / gt_labels.numel()

    end_points['label_loss'] = label_loss

    end_points['label_acc'] = acc
    return label_loss,end_points


