""" Loss functions on unlabeled data

Written by Yezhen Cong, 2020
"""


import numpy as np
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules import loss
FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness
MAX_NUM_OBJ = 64

def get_unlabeled_loss(end_points, ema_end_points):
    labeled_num = torch.nonzero(end_points['supervised_mask']).squeeze(1).shape[0]
 
    ema_point_feature = ema_end_points['pred_feature'][labeled_num:]         #3*20000*2
    pred_feature = end_points['pred_feature'][labeled_num:]                      #3*20000*2


    pred_feature = torch.softmax(pred_feature,-1)                  #3*20000*5
            #
    pred_value, pred_class = torch.max(pred_feature,-1)         #3*20000    3*20000

    pred_value = pred_value.view(pred_value.numel(),-1)       #60000*1
    pred_class = pred_class.view(pred_class.numel(),-1)       #60000*1
    x = torch.ones(pred_value.numel(),1).cuda()
    y = torch.zeros(pred_value.numel(),1).cuda()

    index = torch.where(pred_value>0.8,x,y).squeeze(1)  #1001010111001001001001001010


    new_pred_class = pred_class[torch.nonzero(index)].squeeze(1)

    ema_point_feature = ema_point_feature.view(pred_class.numel(),-1)[torch.nonzero(index)].squeeze(1)
    criterion = nn.CrossEntropyLoss()   #torch的预测方法是，输入那个点的特征n*c玩意
    unlabel_loss = criterion(ema_point_feature.view(new_pred_class.numel(), -1), new_pred_class.view(-1))
    end_points['unlabel_loss'] = unlabel_loss
    return unlabel_loss,end_points

