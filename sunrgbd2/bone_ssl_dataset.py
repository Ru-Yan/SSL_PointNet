""" Labeled and unlabeled dataset for 3DIoUMatch

Author: Zhao Na, 2019
Modified by Yezhen Cong, 2020
"""

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import sys
import random
import torch.utils.data as data
import numpy as np
import os
import pdb
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.pc_util import rotz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class BoneLabeledDataset(Dataset):
    def __init__(self, num_points, folder='LabeledData', sample=True, center=True, weakaugment=None, return_filename=False, orig_first=False, return_piece_idxs=False, return_point_idxs=False):
        super().__init__()
        self.num_points = num_points # 如果orig_first为true, 那么只使用num_points的数据文件
        self.data_dir = os.path.join(BASE_DIR, folder)
        self.all_files = list(filter(lambda x: x.endswith('txt'), os.listdir(self.data_dir)))
        print('\tget {} label data'.format(len(self.all_files)))
        data_min = np.full((6,), np.inf, dtype=np.float32)#无法向量时是3
        data_max = np.full((6,), -np.inf, dtype=np.float32)#无法向量时是3
        self.all_files.sort()
        for filename in self.all_files:
            print(filename)
            pointcloud = np.loadtxt(os.path.join(self.data_dir, filename)).astype(np.float32)

            pointcloud = pd.DataFrame(pointcloud)
            pointcloud = pointcloud.dropna()
            pointcloud = pointcloud.values

            # print('是否有nan:', np.isnan(pointcloud).any())

            if pointcloud.shape[1] == 7:#初始是7
                # includes piece index
                pointcloud = pointcloud[:, :6]#初始是6
            data_min = np.min([np.min(pointcloud, axis=0), data_min], axis=0)
            data_max = np.max([np.max(pointcloud, axis=0), data_max], axis=0)
        self.data_min = data_min # 数据最大值
        self.data_max = data_max # 数据最小值
        self.data_range = self.data_max - self.data_min + 1e-6 # range
        self.sample = sample
        self.center = center # 数据是否尽心中心化
        self.weakaugment = weakaugment
        self.return_filename = return_filename
        self.orig_first = orig_first
        self.return_piece_idxs = return_piece_idxs
        self.return_point_idxs = return_point_idxs

    def __getitem__(self, idx):
        # load data
        label=[]
        filename = self.all_files[idx]
        pointcloud = np.loadtxt(os.path.join(self.data_dir, filename)).astype(np.float32)
        if pointcloud.shape[1] == 7:
            # 读取label
            label = np.uint8(pointcloud[:, 6])
            pointcloud = pointcloud[:, :6]

        if self.center: # 数据中心化操作
            pointcloud[:, :3] = pointcloud[:, :3] - np.mean(pointcloud[:, :3], axis=0)
        # sample data
        if self.sample:
            raw_num_points = pointcloud.shape[0] # 当前数据文件中点的总数量   15555个点
            if self.orig_first:
                pt_idxs = np.array([i for i in range(raw_num_points)])     #array([1-15555])       10000 点
                if self.num_points > raw_num_points: # 如果设定的num_points大于文件中点的数量，那么进行随机化填充
                    pt_idxs = np.concatenate([pt_idxs, np.random.randint(0, high=raw_num_points, size=self.num_points-raw_num_points)])
            else:
                pt_idxs = np.random.randint(0, high=raw_num_points, size=self.num_points)   #pt_idxs(0,)
            pointcloud = pointcloud[pt_idxs]
            label = label[pt_idxs]
        # scale data数据标准化
        m = np.max(np.sqrt(np.sum(pointcloud[:, :3] ** 2, axis=1)))
        pointcloud[:, :3] = pointcloud[:, :3] / m
        # convert data
        current_points = pointcloud[:, :6].copy()#初始是6
        if self.weakaugment is not None:
            if np.random.random() > 0.5:
                current_points[:, 0] = -1 * current_points[:, 0]   #第一维度乘以 
                rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
                rot_mat = rotz(rot_angle)
                current_points[:, 0:3] = np.dot(current_points[:, 0:3], np.transpose(rot_mat))
        return_values = [current_points, label]
        if self.return_filename:
            return_values.append(filename)
        # if self.return_piece_idxs:
        #     return_values.append(piece_idxs)
        # if self.return_point_idxs:
        #     return_values.append(pt_idxs)


        ret_dict = {}
        ret_dict['point_clouds'] = current_points
        ret_dict['labels']=label
        ret_dict['supervised_mask'] = np.array(1).astype(np.int64)
        return ret_dict

    def __len__(self):
        return len(self.all_files)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass
class BoneUnlabeledDataset(Dataset):
    def __init__(self, num_points, folder='UnLabeledData', sample=True, center=True, strongaugment=None, return_filename=False, orig_first=False, return_piece_idxs=False, return_point_idxs=False):
        super().__init__()
        self.num_points = num_points # 如果orig_first为true, 那么只使用num_points的数据文件
        self.data_dir = os.path.join(BASE_DIR, folder)
        self.all_files = list(filter(lambda x: x.endswith('txt'), os.listdir(self.data_dir)))
        print('\tget {} unlabel data'.format(len(self.all_files)))
        data_min = np.full((6,), np.inf, dtype=np.float32)#无法向量时是3
        data_max = np.full((6,), -np.inf, dtype=np.float32)#无法向量时是3
        self.all_files.sort()
        for filename in self.all_files:
            print(filename)
            pointcloud = np.loadtxt(os.path.join(self.data_dir, filename)).astype(np.float32)

            pointcloud = pd.DataFrame(pointcloud)
            pointcloud = pointcloud.dropna()
            pointcloud = pointcloud.values

            # print('是否有nan:', np.isnan(pointcloud).any())

            if pointcloud.shape[1] == 7:#初始是7
                # includes piece index
                pointcloud = pointcloud[:, :6]#初始是6
            data_min = np.min([np.min(pointcloud, axis=0), data_min], axis=0)
            data_max = np.max([np.max(pointcloud, axis=0), data_max], axis=0)
        self.data_min = data_min # 数据最大值
        self.data_max = data_max # 数据最小值
        self.data_range = self.data_max - self.data_min + 1e-6 # range
        self.sample = sample
        self.center = center # 数据是否尽心中心化
        self.strongaugment = strongaugment
        self.return_filename = return_filename
        self.orig_first = orig_first
        self.return_piece_idxs = return_piece_idxs
        self.return_point_idxs = return_point_idxs

    def __getitem__(self, idx):
        # load data
        label=[]
        filename = self.all_files[idx]
        pointcloud = np.loadtxt(os.path.join(self.data_dir, filename)).astype(np.float32)
        if pointcloud.shape[1] == 7:
            # 读取label
            label = np.uint8(pointcloud[:, 6])
            pointcloud = pointcloud[:, :6]

        if self.center: # 数据中心化操作
            pointcloud[:, :3] = pointcloud[:, :3] - np.mean(pointcloud[:, :3], axis=0)
        # sample data
        if self.sample:
            raw_num_points = pointcloud.shape[0] # 当前数据文件中点的总数量   15555个点
            if self.orig_first:
                pt_idxs = np.array([i for i in range(raw_num_points)])     #array([1-15555])       10000 点
                if self.num_points > raw_num_points: # 如果设定的num_points大于文件中点的数量，那么进行随机化填充
                    pt_idxs = np.concatenate([pt_idxs, np.random.randint(0, high=raw_num_points, size=self.num_points-raw_num_points)])
            else:
                pt_idxs = np.random.randint(0, high=raw_num_points, size=self.num_points)   #pt_idxs(0,)
            pointcloud = pointcloud[pt_idxs]
            label = label[pt_idxs]
        # scale data数据标准化
        m = np.max(np.sqrt(np.sum(pointcloud[:, :3] ** 2, axis=1)))
        pointcloud[:, :3] = pointcloud[:, :3] / m
        # convert data
        current_points = pointcloud[:, :6].copy()#初始是6
        if self.strongaugment is not None:
            current_points = self.strongaugment(current_points)

        return_values = [current_points, label]
        if self.return_filename:
            return_values.append(filename)
        # if self.return_piece_idxs:
        #     return_values.append(piece_idxs)
        # if self.return_point_idxs:
        #     return_values.append(pt_idxs)


        ret_dict = {}
        ret_dict['point_clouds'] = current_points
        ret_dict['labels']=label
        ret_dict['supervised_mask'] = np.array(0).astype(np.int64)
        return ret_dict

    def __len__(self):
        return len(self.all_files)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass

class BoneTestDataset(Dataset):
    def __init__(self, num_points, folder='TestData', sample=True, center=True, transforms=None, return_filename=False, orig_first=False, return_piece_idxs=False, return_point_idxs=False):
        super().__init__()
        self.num_points = num_points # 如果orig_first为true, 那么只使用num_points的数据文件
        self.data_dir = os.path.join(BASE_DIR, folder)
        self.all_files = list(filter(lambda x: x.endswith('txt'), os.listdir(self.data_dir)))
        print('\tget {} testlabel data'.format(len(self.all_files)))
        data_min = np.full((6,), np.inf, dtype=np.float32)#无法向量时是3
        data_max = np.full((6,), -np.inf, dtype=np.float32)#无法向量时是3
        self.all_files.sort()
        for filename in self.all_files:
            print(filename)
            pointcloud = np.loadtxt(os.path.join(self.data_dir, filename)).astype(np.float32)

            pointcloud = pd.DataFrame(pointcloud)
            pointcloud = pointcloud.dropna()
            pointcloud = pointcloud.values

            # print('是否有nan:', np.isnan(pointcloud).any())

            if pointcloud.shape[1] == 7:#初始是7
                # includes piece index
                pointcloud = pointcloud[:, :6]#初始是6
            data_min = np.min([np.min(pointcloud, axis=0), data_min], axis=0)
            data_max = np.max([np.max(pointcloud, axis=0), data_max], axis=0)
        self.data_min = data_min # 数据最大值
        self.data_max = data_max # 数据最小值
        self.data_range = self.data_max - self.data_min + 1e-6 # range
        self.sample = sample
        self.center = center # 数据是否尽心中心化
        self.transforms = transforms
        self.return_filename = return_filename
        self.orig_first = orig_first
        self.return_piece_idxs = return_piece_idxs
        self.return_point_idxs = return_point_idxs

    def __getitem__(self, idx):
        # load data
        label=[]
        filename = self.all_files[idx]
        pointcloud = np.loadtxt(os.path.join(self.data_dir, filename)).astype(np.float32)
        if pointcloud.shape[1] == 7:
            # 读取label
            label = np.uint8(pointcloud[:, 6])
            pointcloud = pointcloud[:, :6]
        pts_back = pointcloud.copy()
        if self.center: # 数据中心化操作
            pointcloud[:, :3] = pointcloud[:, :3] - np.mean(pointcloud[:, :3], axis=0)
        # sample data
        if self.sample:
            raw_num_points = pointcloud.shape[0] # 当前数据文件中点的总数量   15555个点
            if self.orig_first:
                pt_idxs = np.array([i for i in range(raw_num_points)])     #array([1-15555])       10000 点
                if self.num_points > raw_num_points: # 如果设定的num_points大于文件中点的数量，那么进行随机化填充
                    pt_idxs = np.concatenate([pt_idxs, np.random.randint(0, high=raw_num_points, size=self.num_points-raw_num_points)])
            else:
                pt_idxs = np.random.randint(0, high=raw_num_points, size=self.num_points)   #pt_idxs(0,)
            pointcloud = pointcloud[pt_idxs]
            label = label[pt_idxs]
            pts_back = pts_back[pt_idxs]
        # scale data数据标准化
        m = np.max(np.sqrt(np.sum(pointcloud[:, :3] ** 2, axis=1)))
        pointcloud[:, :3] = pointcloud[:, :3] / m
        # convert data
        current_points = pointcloud[:, :6].copy()#初始是6
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return_values = [current_points, label]
        if self.return_filename:
            return_values.append(filename)
        # if self.return_piece_idxs:
        #     return_values.append(piece_idxs)
        # if self.return_point_idxs:
        #     return_values.append(pt_idxs)


        ret_dict = {}
        ret_dict['raw_point_clouds'] = pts_back
        ret_dict['point_clouds'] = current_points
        ret_dict['labels']=label
        ret_dict['supervised_mask'] = np.array(1).astype(np.int64)
   
        return ret_dict

    def __len__(self):
        return len(self.all_files)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


    
if __name__ == '__main__':
    dataset = BoneLabeledDataset(num_points=20000)
    y= dataset[0]

    a=input()
