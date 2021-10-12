from __future__ import print_function
import argparse
import torch.optim as optim
import models

import os
from lib.LinearAverage_2 import LinearAverage
from test import NN, kNN_DA, recompute_memory
from return_dataset import return_dataset_selfsup, set_model_self
import logging
from utils_dh import setup_logging
import sys

from custom_function.selfsup import *

torch.backends.cudnn.benchmark=True


# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--eta_b', type=float, default=1.0, metavar='ETAB',
                    help='eta b')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./output',
                    help='dir to save checkpoint') 
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='Real', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='Clipart', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default='office_home', choices=['office', 'office_home', 'cub'],
                    help='the name of dataset, multi is large scale dataset')
parser.add_argument('--split', type=int, default=0, metavar='N',
                    help='which split to use')
parser.add_argument('--epoch', type=int, default=0, metavar='N',
                    help='how many labeled examples to use for target domain')
parser.add_argument('--pretrained_batch', type=int, default=32, metavar='N',
                    help='how many labeled examples to use for target domain')

parser.add_argument('--low_dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='M', help='batch_size')
parser.add_argument('--nce-k', default=0, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

parser.add_argument('--n_neighbor', default=700, type=int,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--temp2', default=1.0, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--lambda_value', default=1.0, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='validation phase')
parser.add_argument('--training_da', action='store_false', default=True,
                    help='validation phase')

parser.add_argument('--imagenet', action='store_true', default=False,
                    help='validation phase')

parser.add_argument('--DC', action='store_true', default=False,
                    help='validation phase')
parser.add_argument('--instance', action='store_true', default=True,
                    help='validation phase')
parser.add_argument('--t2s', action='store_true', default=True,
                    help='validation phase')
parser.add_argument('--s2t', action='store_true', default=True,
                    help='validation phase')

parser.add_argument('--bs1', type=int, default=2, metavar='N',
                    help='how many labeled examples to use for target domain')
parser.add_argument('--bs2', type=int, default=2, metavar='N',
                    help='how many labeled examples to use for target domain')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

source_loader, target_loader_unl, val_loader, _, class_list = return_dataset_selfsup(args, batch_size=args.batch_size)

use_gpu = torch.cuda.is_available()


source_len = source_loader.dataset.__len__()
target_len = val_loader.dataset.__len__()



if target_len > source_len:
    args.bs2 = int(args.bs2 * (target_len/source_len))
else:
    args.bs1 = int(args.bs1 * (source_len/target_len))


name = 'CDS_{}_{}_{}'.format(args.dataset, args.source, args.target)
name_weight = 'CDS_{}_{}_{}'.format(args.dataset, args.source, args.target)

if args.dataset == 'visda':

    name_weight = name_weight + '_bs1_4_bs2_2visda'


if args.scratch:
    args.multi *= 10
    args.lr *= 10
    # args.lr_da *= 10

setup_logging(name)
logging.info(' '.join(sys.argv))
logging.info('dataset %s source %s target %s split%d' % (args.dataset, args.source, args.target, args.split))
# torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
params = []
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if args.net == 'resnet50':
    inc = 2048
    net = models.__dict__['resnet50'](pretrained=not args.scratch, low_dim=args.low_dim)

lemniscate_s = LinearAverage(args.low_dim, source_loader.dataset.__len__() , args.nce_t, args.nce_m)
lemniscate_t = LinearAverage(args.low_dim, target_loader_unl.dataset.__len__(), args.nce_t, args.nce_m)
ndata = source_loader.dataset.__len__() + target_loader_unl.dataset.__len__()
lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

logging.info('source data len : {}'.format(source_loader.dataset.__len__()))
logging.info('target data len : {}'.format(target_loader_unl.dataset.__len__()))


net.cuda()
lemniscate_s.cuda()
lemniscate_t.cuda()
lemniscate.cuda()
######


logging.info(args.lambda_value)
net.train()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

## CDS


if os.path.exists(args.checkpath)==False:
    os.mkdir(args.checkpath)


for epoch in range(0,7):

    logging.info('epoch: start:{}'.format(epoch))
    train_selfsup_only(epoch, args, net, name_weight, device, lemniscate_s,
                      lemniscate_t, lemniscate, optimizer, save_load=False)


