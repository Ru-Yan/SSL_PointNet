# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def get_loss(end_points):
    supervised_mask = end_points['supervised_mask']
    supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()   #假设label为2 [0,1]

    gt_labels=end_points['labels'][supervised_inds, ...]         #2*20000
    gt_labels = gt_labels.long()

    pred_feature=end_points['pred_feature'][supervised_inds, ...]           #2*20000*2


    _, classes = torch.max(pred_feature, -1)

    acc = (classes == gt_labels).float().sum() / gt_labels.numel()

    end_points['pre_labels'] = classes
    end_points['test_label_acc'] = acc
    
    return end_points
