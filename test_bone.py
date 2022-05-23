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

from models.pointnet2_ssg_sem import Pointnet2MSG#    加载网络核心模型

BASE_DIR = os.path.dirname(os.path.abspath(__file__))    #获取路径
ROOT_DIR = BASE_DIR          
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler        #优化器加载 
from utils.tf_visualizer import Visualizer as TfVisualizer    #tf可视化
from models.loss_helper_labeled import get_labeled_loss
from models.loss_helper_unlabeled import get_unlabeled_loss
from models.loss_helper import get_loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Pointnet2SSG', help='Model file name.')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet.')

parser.add_argument('--log_dir', default='./temp', help='Dump dir to save model checkpoint')
parser.add_argument('--detector_checkpoint', default='/home/csc/桌面/SSL-pointnet2/temp/checkpoint.tar')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number')
parser.add_argument('--use_vector', action='store_true', help='Use vector input.')
parser.add_argument('--max_epoch', type=int, default=1001, help='Epoch to run')
parser.add_argument('--batch_size', default='2,3', help='Batch Size during training, labeled + unlabeled')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs)')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay')
parser.add_argument('--lr_decay_steps', default='400, 600, 800, 900',
                    help='When to decay the learning rate (in epochs)')
parser.add_argument('--lr_decay_rates', default='0.3, 0.3, 0.1, 0.1',
                    help='Decay rates for lr decay')
parser.add_argument('--ema_decay',  type=float,  default=0.999, metavar='ALPHA',
                    help='ema variable decay rate')
parser.add_argument('--unlabeled_loss_weight', type=float, default=2.0, metavar='WEIGHT',
                    help='use unlabeled loss with given weight')
parser.add_argument('--print_interval', type=int, default=25, help='batch interval to print loss')
parser.add_argument('--eval_interval', type=int, default=25, help='epoch interval to evaluate model')
parser.add_argument('--save_interval', type=int, default=200, help='epoch interval to save model')
parser.add_argument('--resume', action='store_true', help='resume training instead of just loading a pre-train model')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--conf_thresh', type=float, default=0.05)
parser.add_argument('--cls_number', type=int, default=10, help='Please Set classification number')
FLAGS = parser.parse_args()
print(FLAGS)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
print('\n************************** GLOBAL CONFIG BEG **************************')
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

num_input_channel = 3
# Init datasets and dataloaders
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd.bone_ssl_dataset import BoneTestDataset
   
                                                            
    TEST_DATASET = BoneTestDataset(num_points=NUM_POINT)
   




else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)


TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1,
                             shuffle=False, num_workers=1, worker_init_fn=my_worker_init_fn)

def create_model(ema=False):
    model = Pointnet2MSG(num_classes=CLASS_NUMBER, input_channels=0, use_xyz=True)   #有法向量channel为3，否则为0
    if ema:
        for param in model.parameters():
            param.detach_()
    return model



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = create_model()
ema_detector = create_model(ema=True)
detector.to(device)
ema_detector.to(device)
test_detector_criterion = get_loss

# Load the Adam optimizer
optimizer = optim.Adam(detector.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)#     优化网络参数

# Load checkpoint if there is any   加载训练模型

checkpoint = torch.load('temp/checkpoint_400.tar')
pretrained_dict = checkpoint['model_state_dict']               # 网络参数放入 当中


model_dict = detector.state_dict()                            #更新现在用的pointnet2参数
                           
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)

detector.load_state_dict(model_dict)
model_dict = ema_detector.state_dict()
model_dict.update(pretrained_dict)
ema_detector.load_state_dict(model_dict)    

epoch_ckpt = checkpoint['epoch']
print("Loaded pointnet2 checkpoint %s (epoch: %d)" % (FLAGS.detector_checkpoint, epoch_ckpt))
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(detector, bn_lambda=bn_lbmd, last_epoch=-1)



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
    raw_point_clouds = end_points['raw_point_clouds'].squeeze(0).cpu().numpy()
    labels = end_points['pre_labels'].cpu().numpy()
    Bone = np.concatenate([raw_point_clouds,labels.reshape(20000,1)],axis=-1)
    np.savetxt('results/pre_bone.txt',Bone,fmt='%.06f')
    print('预测准确率为:',end_points['test_label_acc'].cpu().numpy())


