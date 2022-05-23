""" Training (stage 2) function for 3DIoUMatch

Written by: Yezhen Cong, 2020
Based on: VoteNet and SESS
"""
import argparse
import os

import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader #pytorch 数据处理，后期根据batch_size 封装成tensor
from torchvision import transforms#torchvision.transforms包含resize、crop等常见的data augmentation
from models.pointnet2_ssg_sem import Pointnet2MSG#    加载网络核心模型
import pointnet2.data_utils as d_utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))    #获取路径
ROOT_DIR = BASE_DIR          
#sys.path.append(os.path.join(ROOT_DIR, 'utils'))
#sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
#sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler        #优化器加载 
from utils.tf_visualizer import Visualizer as TfVisualizer    #tf可视化
from models.loss_helper_labeled import get_labeled_loss
from models.loss_helper_unlabeled import get_unlabeled_loss
from models.loss_helper import get_loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Pointnet2SSG', help='Model file name.')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet.')

parser.add_argument('--log_dir', default='./temp', help='Dump dir to save model checkpoint')
parser.add_argument('--detector_checkpoint', default='None', help='/home/csc/桌面/SSL-pointnet2/temp/checkpoint.tar')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number')
parser.add_argument('--max_epoch', type=int, default=2000, help='Epoch to run')
parser.add_argument('--batch_size', default='1,3', help='Batch Size during training, labeled + unlabeled')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs)')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay')
parser.add_argument('--lr_decay_steps', default='400,800,1200,1600',
                    help='When to decay the learning rate (in epochs)')
parser.add_argument('--lr_decay_rates', default='0.3 ,0.3,0.1, 0.1',
                    help='Decay rates for lr decay')
parser.add_argument('--ema_decay',  type=float,  default=0.999, metavar='ALPHA',
                    help='ema variable decay rate')
parser.add_argument('--unlabeled_loss_weight', type=float, default=18.0, metavar='WEIGHT',
                    help='use unlabeled loss with given weight')
parser.add_argument('--print_interval', type=int, default=5, help='batch interval to print loss')
parser.add_argument('--eval_interval', type=int, default=1, help='epoch interval to evaluate model')
parser.add_argument('--save_interval', type=int, default=200, help='epoch interval to save model')
parser.add_argument('--resume', default=False, help='resume training instead of just loading a pre-train model')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--conf_thresh', type=float, default=0.05)
parser.add_argument('--cls_number', type=int, default=10, help='Please Set classification number')
FLAGS = parser.parse_args()
print(FLAGS)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
print('\n************************** GLOBAL CONFIG BEG **************************')
print(torch.cuda.is_available())
batch_size_list = [int(x) for x in FLAGS.batch_size.split(',')]
BATCH_SIZE = batch_size_list[0] + batch_size_list[1]
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
CLASS_NUMBER = FLAGS.cls_number
assert(len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
if not FLAGS.eval:
    PERFORMANCE_FOUT = open(os.path.join(LOG_DIR, 'best.txt'), 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)  #这边定义了一个函数输入      把训练的参数过程写入txt


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

num_input_channel = 0
# Init datasets and dataloaders
if FLAGS.dataset == 'sunrgbd':


    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd.bone_ssl_dataset import BoneLabeledDataset, BoneUnlabeledDataset, BoneTestDataset
    LABELED_DATASET = BoneLabeledDataset(num_points=NUM_POINT,weakaugment=True,center=True)                                               
    UNLABELED_DATASET = BoneUnlabeledDataset(num_points=NUM_POINT,strongaugment=transforms, center=True)
    TEST_DATASET = BoneTestDataset(num_points=NUM_POINT)
   




else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)

LABELED_DATALOADER = DataLoader(LABELED_DATASET, batch_size=batch_size_list[0],
                              shuffle=True, num_workers=batch_size_list[0], worker_init_fn=my_worker_init_fn)
UNLABELED_DATALOADER = DataLoader(UNLABELED_DATASET, batch_size=batch_size_list[1],
                              shuffle=True, num_workers=batch_size_list[1]//2, worker_init_fn=my_worker_init_fn)

TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1,
                             shuffle=False, num_workers=1, worker_init_fn=my_worker_init_fn)

def create_model(ema=False):
    model = Pointnet2MSG(num_classes=CLASS_NUMBER, input_channels=num_input_channel, use_xyz=True)   #有法向量channel为3，否则为0
    if ema:
        for param in model.parameters():
            param.detach_()
    return model



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = create_model()
ema_detector = create_model(ema=True)
detector.to(device)
ema_detector.to(device)
train_labeled_criterion = get_labeled_loss
train_unlabeled_criterion = get_unlabeled_loss
test_detector_criterion = get_loss

# Load the Adam optimizer
optimizer = optim.Adam(detector.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)#     优化网络参数

# Load checkpoint if there is any   加载训练模型
if FLAGS.detector_checkpoint is not None and os.path.isfile(FLAGS.detector_checkpoint):
    print("==============================load checkpoint==================================")
    checkpoint = torch.load(FLAGS.detector_checkpoint)
    print(checkpoint)
    pretrained_dict = checkpoint['model_state_dict']               # 网络参数放入 当中

    ########
    model_dict = detector.state_dict()                            #更新现在用的pointnet2参数
    # 1. filter out unnecessary keys                              #
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    detector.load_state_dict(model_dict)
    model_dict = ema_detector.state_dict()
    model_dict.update(pretrained_dict)
    ema_detector.load_state_dict(model_dict)    
    ########

    if FLAGS.resume:
        print("===========================resume=======================")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    # detector.load_state_dict(pretrained_dict)
    # ema_detector.load_state_dict(pretrained_dict)
    epoch_ckpt = checkpoint['epoch']
    print("========Loaded pointnet2 checkpoint %s (epoch: %d)=================" % (FLAGS.detector_checkpoint, epoch_ckpt))
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
# inherited this from VoteNet and SESS
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(detector, bn_lambda=bn_lbmd, last_epoch=-1)
if FLAGS.resume:
    bnm_scheduler.step(start_epoch)


def get_current_lr(epoch):
    # stairstep update
    lr = BASE_LEARNING_RATE            #起初为0.002   衰减为400 600 800 900 0.002*0.3 
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)                 #获取当前的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr         #调整学习率函数，lr学习率传递给优化器


# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(LOG_DIR, 'train')
TEST_VISUALIZER = TfVisualizer(LOG_DIR, 'test')


# Used for Pseudo box generation and AP calculation
CONFIG_DICT = {'unlabeled_batch_size': batch_size_list[1], 'dataset': FLAGS.dataset,

               'remove_empty_box': False, 'use_3d_nms': True, 'nms_iou': 0.25,
               'use_old_type_nms': False, 'cls_nms': True, 

               'per_class_proposal': True, 'conf_thresh': FLAGS.conf_thresh,

               'obj_threshold': 0.9, 'cls_threshold': 0.9,
               'use_lhs': True, 'iou_threshold': 0.25,

               'use_unlabeled_obj_loss': False, 'use_unlabeled_vote_loss': False, 'vote_loss_size_factor': 1.0,

               'samecls_match': False}

for key in CONFIG_DICT.keys():
    if key != 'dataset_config':
        log_string(key + ': ' + str(CONFIG_DICT[key]))

print('************************** GLOBAL CONFIG END **************************')
# ------------------------------------------------------------------------- GLOBAL CONFIG END


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)    #  min(0.5,0.99)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)    #参数传递x.mul_(y),_有了个赋值过程

#    Z-teacher=Z-teacher*a+(1-a)*Z-student
def tb_name(key):      # 这玩意是tensorboard的显示界面   acc放一起，l含loss放一起
    if 'loss' in key:
        return 'loss/' + key
    elif 'acc' in key:
        return 'acc/' + key
    elif 'ratio' in key:
        return 'ratio/' + key
    elif 'value' in key:
        return 'value/' + key
    else:
        return 'other/' + key


def train_one_epoch(global_step):
    stat_dict = {}  # 定义一个字典，数据统计
    adjust_learning_rate(optimizer, EPOCH_CNT)  #调整学习率
    bnm_scheduler.step()  # decay BN momentum
    detector.train()  # set model to training mode
    ema_detector.train()

    unlabeled_dataloader_iterator = iter(UNLABELED_DATALOADER) #iter函数了解一下  未标记数据加载迭代 iter创建可迭代对象

    for batch_idx, batch_data_label in enumerate(LABELED_DATALOADER):
        try:
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)   #无标签点云数据
        except StopIteration:
            unlabeled_dataloader_iterator = iter(UNLABELED_DATALOADER)
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)

        for key in batch_data_unlabeled:                 #这边进行了一个循环 把没有标签的数据的，和标签的数据拼接起来
            if type(batch_data_unlabeled[key]) == list:
                batch_data_label[key] = torch.cat([batch_data_label[key]] + batch_data_unlabeled[key], dim=0)#.to(device)
            else:
                batch_data_label[key] = torch.cat((batch_data_label[key], batch_data_unlabeled[key]), dim=0)#.to(device)

        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)     #把字典里面存放的值全部放到gpu中 device=cuda

        inputs = {'point_clouds': batch_data_label['point_clouds']}  #网络输入为3*40000*4

        optimizer.zero_grad()
        with torch.no_grad():
            ema_end_points = ema_detector(inputs['point_clouds'])
            
        end_points = detector(inputs['point_clouds'])

        # Compute loss and gradients, update parameters.     计算loss和梯度 更新参数
        for key in batch_data_label:
            assert(key not in end_points)          #endpoint没有的，加进去   endpoint里面存的最终的有标签的初始的信息，和最终的预测信息
            end_points[key] = batch_data_label[key]

        detection_loss, end_points = train_labeled_criterion(end_points) # 利用endpoint计算有标签loss

        unlabeled_loss, end_points = train_unlabeled_criterion(end_points, ema_end_points) #利用end_point 和ema_end_point,计算无标签loss

        loss = detection_loss + unlabeled_loss * FLAGS.unlabeled_loss_weight


        end_points['loss'] = loss
        loss.backward()            #反向传播    

        optimizer.step()          #优化起进行一次
        global_step += 1
        update_ema_variables(detector, ema_detector, FLAGS.ema_decay, global_step)      #参数更新一次

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key or 'value' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()        #先把所有值加起来，后面求均值

        batch_interval = FLAGS.print_interval
        if (batch_idx + 1) % batch_interval == 0: #idx为24时候，数据加载24次，反向传播24次，24个key值，打印一下
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            TRAIN_VISUALIZER.log_scalars({tb_name(key): stat_dict[key] / batch_interval for key in stat_dict},
                                         (EPOCH_CNT * len(LABELED_DATALOADER) + batch_idx) * BATCH_SIZE)     #存进TF可视化工具 5*24
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0                                                     #归0 进行下一轮训练

    return global_step


AP_IOU_THRESHOLDS = [0.25, 0.5]        #阈值   0.25，0.5
BEST_MAP = [0.0, 0.0]


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics

    detector.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = detector(inputs['point_clouds'])

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        end_points = test_detector_criterion(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key or 'value' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    # Log statistics
    TEST_VISUALIZER.log_scalars({tb_name(key): stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
                                (EPOCH_CNT + 1) * len(LABELED_DATALOADER) * BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
def train():
    global EPOCH_CNT     #申明全局变量 
    global start_epoch   #一次epoch，迭代数目=N=训练样本数/batch_size
    global_step = 0
    loss = 0
    EPOCH_CNT = 0          #CNT
    global BEST_MAP  
    start_from = 0
 
    for epoch in range(start_from, MAX_EPOCH):   #开始第一次训练
        EPOCH_CNT = epoch                         #EPOCH_CNT不知道干嘛的
        log_string('\n**** EPOCH %03d, STEP %d ****' % (epoch, global_step))       #
        log_string("Current epoch: %d, obj threshold = %.3f & cls threshold = %.3f" % (epoch, CONFIG_DICT['obj_threshold'], CONFIG_DICT['cls_threshold']))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))    
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now())) #获取时间并输出

        # np.random.get_state()[1][0] 在某种状态下生成的随机数相同
        # 与测试
        np.random.seed()             #生成固定随机数   开始第一轮训练
        global_step = train_one_epoch(global_step)    # 输入轮次0
        map = 0.0
        if EPOCH_CNT > 0 and EPOCH_CNT % FLAGS.eval_interval == 0:    #25轮都会进行评估
            evaluate_one_epoch()
        # save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss
                     }
        print('模型参数更新')
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = detector.module.state_dict()
            save_dict['ema_model_state_dict'] = ema_detector.module.state_dict()
        except:
            save_dict['model_state_dict'] = detector.state_dict()
            save_dict['ema_model_state_dict'] = ema_detector.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))    

        if EPOCH_CNT % FLAGS.save_interval == 0:          #每间隔200轮保存一次
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),    #优化器状态
                         'loss': loss
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = detector.module.state_dict()
                save_dict['ema_model_state_dict'] = ema_detector.module.state_dict()
            except:
                save_dict['model_state_dict'] = detector.state_dict()
                save_dict['ema_model_state_dict'] = ema_detector.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_%d.tar' % EPOCH_CNT))   #两个模型的网络参数，优化器的当前状态


if __name__ == '__main__':
    train()
