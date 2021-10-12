import logging
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


path_prefix = ['/home/ubuntu/', '']

color = np.array([(0,0,0),(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255), #magenta
        (192,192,192), #silver
        (128,128,128), #gray
        (128,0,0) ,#maroon
        (128,128,0) ,#olive
        (0,128,0) ,#green
        (128,0,128), # purple
        (0,128,128) , # teal
        (65,105,225) , #royal blue
        (255,250,205) , #lemon chiffon
        (255,20,147) , #deep pink
        (218,112,214) , #orchid]
        (135,206,250) , #light sky blue
        (127,255,212),  #aqua marine
        (0,255,127) , #spring green
        (255,215,0) , #gold
        (165,42,42) , #brown
        (148,0,211) , #violet
        (210,105,30) , # chocolate
        (244,164,96),  # sandy brown
        (240,255,240),  #honeydew
        (112,128,144), (64,224,208) ,(100,149,237) ,(30,144,255),(221,160,221),
        (205,133,63),(255,240,245),(255,255,240),(255,165,0),(255,160,122),(205,92,92),(240,248,255)])

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


def set_deviceid(id=[0]):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

def setup_logging(file_name, pre_fix=''):
    import datetime
    import logging
    if not os.path.isdir('./logging'):
        os.makedirs('./logging')
    exp_dir = os.path.join('./logging/', file_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_fn = os.path.join(exp_dir, pre_fix+"LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
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



def load_model(model, path, layer_except='up'):
    #
    m_dict = torch.load(path)['state_dict']

    pretrained_dict = {k: v for k, v in m_dict.items() if k.find(layer_except) == -1}

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model