import os
import torch
import shutil
import numpy as np
from torch.optim import Optimizer
from scipy import stats
import pdb
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize
def return_density(x,bandwidth=0.2):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(x)
    # score_samples() returns the log-likelihood of the samples
    return kde_skl

def return_density2(data):
    kernel = stats.gaussian_kde(np.transpose(data))
    return kernel
def importance(source,target):
    source = normalize(source)
    target = normalize(target)
    source_kernel = return_density(source)
    target_kernel = return_density(target)
    #pdb.set_trace()
    source_density = np.exp(source_kernel.score_samples(source))
    target_density = np.exp(target_kernel.score_samples(source))
    importance = target_density/source_density
    return importance
def imp_loss(source,target,source_acc):

    weight_imp = importance(source,target)

    return np.mean(weight_imp*source_acc)


class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))