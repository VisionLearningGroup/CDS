import logging
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(args):

    if args.dataset == 'office_home':

        args.s_dset_path = '../data/office_home/new_images_{}_train_{}_shot_labeled_sp_{}.txt'.format(args.source, args.shot, args.py_seed)

        args.s_dset_path_ul = '../data/office_home/new_images_{}_train_{}_shot_unlabeled_sp_{}.txt'.format(args.source, args.shot,args.py_seed)

        args.t_dset_path = '../data/office_home/labeled_source_images_{}.txt'.format(
            args.target)

        args.val_dset_path = '../data/office_home/labeled_source_images_{}.txt'.format(
            args.target)

        args.name_weight = 'CDS_{}_{}_{}'.format(args.dataset, args.source, args.target)
        args.test_interval = 250


def set_deviceid(id=[0]):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id[0])


def setup_logging(file_name):
    import datetime
    import logging
    if not os.path.isdir('./logging'):
        os.makedirs('./logging')
    exp_dir = os.path.join('./logging/', file_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
    logging.basicConfig(filename=log_fn, filemode='a', level=logging.DEBUG, format='%(message)s')
    # also log into console
    console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.getLogger().setLevel(logging.INFO)
    print('Loging into %s...' % exp_dir)



def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image


def set_config(args):

    if args.dataset == 'office_home':
        args.log_interval = 500

    if args.net == 'resnet50':
        args.batch_size = 32



def load_model(model, path, layer_except='up'):
    #
    m_dict = torch.load(path)['state_dict']

    pretrained_dict = {k: v for k, v in m_dict.items() if k.find(layer_except) == -1}

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model