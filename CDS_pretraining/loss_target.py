import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb
import numpy as np
from torch.autograd import Function

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
def adr(model,feat,T=0.05,k=2,eta=0.05,conf=False):
    #feat_drop = F.dropout(feat,training=model.training)
    out1 = F.softmax(model(feat,reverse=True))
    out2 = F.softmax(model(feat,reverse=True))
    loss = -torch.mean(torch.abs(out1-out2))
    return loss


def entropy(F1,feat,lamda, eta=0):
    if eta == 0:
        out_t1 = F1(feat)
    else:
        out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

def kl_div(out1,out2):
    return torch.mean(torch.sum(out1 * torch.log(out1/out2 + 1e-5),1))


def adentropy(F1,feat,lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


def adentropy_2(F1,feat,lamda, eta=1.0):
    out_t1, feat  = F1.forward_2(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent, feat


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

