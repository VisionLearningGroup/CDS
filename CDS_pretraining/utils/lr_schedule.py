import numpy as np
import pdb


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001, power=0.75, init_lr=0.001,weight_decay=0.0005):

    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    #print(lr)
    i=0
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr * param_lr[i]

        i+=1
    return optimizer

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)



def adjust_learning_rate(optimizer, step, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr

    if step == 20000 or step == 40000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer