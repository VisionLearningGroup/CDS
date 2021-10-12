# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import pdb



def weights_init(m):
    print('weight init')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def adr(model,feat,lambd,T=0.05,k=2,eta=0.05,conf=False):
    feat_drop = F.dropout(feat,training=model.training)
    out1 = F.softmax(model(feat,reverse=True))
    out2 = F.softmax(model(feat_drop,reverse=True))
    loss = -kl_div(out1,out2)#torch.mean(torch.abs(out1-out2))
    return loss

def kl_div(out1,out2):
    return torch.mean(torch.sum(out1 * torch.log(out1/out2 + 1e-5),1))
def virtual_weight(model,feat,lambd,T=0.05,k=2,eta=0.05,conf=False,flip=False):
    w_temp = model.fc3_1.weight
    feat = F.normalize(feat)
    sum = False
    loss = 0
    for i in range(k):
        model.zero_grad()
        w_temp.requires_grad_()
        out_t1 = torch.mm(feat.detach(), w_temp.t() / T)
        out_t1 = F.softmax(out_t1)
        size_t = out_t1.size(0)
        loss_d = torch.sum(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))/size_t
        loss -= loss_d
        loss_d.backward(retain_graph=True)
        #pdb.set_trace()
        w_delta = -w_temp.grad * eta#-F.normalize(w_temp.grad) * torch.norm(w_temp,dim=1).view(w_temp.size(0),1)*eta
        if flip:
            w_delta = - w_delta
        norm_temp = torch.norm(w_temp,dim=1).view(w_temp.size(0),1)
        #pdb.set_trace()
        w_temp_delta = w_delta #+ F.normalize(w_temp)*torch.sum(torch.mm(w_temp,w_delta.t()),1).view(-1,1)
        w_temp = w_temp + w_temp_delta
        #w_temp = F.normalize(w_temp) * norm_temp
        w_temp = Variable(w_temp)
        w_temp.requires_grad_()
        # else:
        #     loss_d = lambd * torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1)
        #     for t in range(loss_d.size(0)):
        #         delta = torch.autograd.grad(loss_d[i], w_temp,retain_graph=True)[0]
        #         w_delta = -delta
        #         w_temp1 = Variable(w_temp+w_delta)
        #         out_tem = F.softmax(torch.mm(feat[i].view(1,-1), w_temp1.t() / T))
        #         if t==0:
        #             out_d = out_tem
        #         else:
        #             out_d = torch.cat([out_d,out_tem],0)
        if i == 0:
            return_loss = -lambd * loss_d.mean()
    out_d = F.softmax(torch.mm(feat, w_temp.t() / T))
    loss = -lambd * torch.sum(torch.sum(out_d * (torch.log(out_d + 1e-5)), 1))/size_t
    return return_loss, loss

def return_virtual_weight(model,feat,T=0.05,k=2,eta=0.05,conf=False):
    w_temp = model.fc3_1.weight
    feat = F.normalize(feat)
    sum = False
    loss = 0
    model.zero_grad()
    w_temp.requires_grad_()
    out_t1 = torch.mm(feat.detach(), w_temp.t() / T)
    out_t1 = F.softmax(out_t1)
    size_t = out_t1.size(0)
    loss_d = torch.sum(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))/size_t
    loss -= loss_d
    loss_d.backward(retain_graph=True)
    w_delta = -w_temp.grad * eta
    w_temp = w_temp + w_delta
    w_temp = Variable(w_temp)
    w_temp.requires_grad_()
    out_d = F.softmax(torch.mm(feat, w_temp.t() / T))
    #loss = -lambd * torch.sum(torch.sum(out_d * (torch.log(out_d + 1e-5)), 1))/size_t
    return out_d

def virtual_weight_vat(model,feat,lambd,T=0.05,k=2,eta=0.1):

    w_temp = model.fc3_1.weight
    pred = F.softmax(torch.mm(feat.detach(), w_temp.t() / T))
    #pdb.set_trace()
    d = F.normalize(Variable(torch.rand(w_temp.size()).cuda())) * 0.1
    d.requires_grad_()
    w_temp = w_temp + d
    feat = F.normalize(feat)
    for i in range(k):
        model.zero_grad()
        out_t1 = torch.mm(feat.detach(), w_temp.t() / T)
        logp_hat = F.log_softmax(out_t1, dim=1)
        adv_distance = F.kl_div(logp_hat, pred.detach())
        loss_d = adv_distance
        loss_d.backward()
        d_delta = -d.grad * eta
        w_temp = w_temp + d_delta
        w_temp = Variable(w_temp)
        w_temp.requires_grad_()
        if i == 0:
            return_loss = lambd * loss_d.detach()
    out_d = F.softmax(torch.mm(feat, w_temp.t() / T))
    logp_hat = F.log_softmax(out_d, dim=1)
    loss_adv = F.kl_div(logp_hat, pred.detach())
    return return_loss, loss_adv
def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p), 1))
def kl_torch(out1,out2):
    return torch.mean(torch.sum(out1*torch.log(out1/out2+1e-5),1))
def mmd(x, y, alpha=[0.01, 0.05, 0.0001, 0.1, 0.000001]):
    loss = 0
    #pdb.set_trace()
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    if x.size(0)/2 == y.size(0):
        ry_t = torch.cat([ry,ry],0)
        ry_t = torch.cat([ry_t,ry_t],1)
        zz = torch.cat([zz,zz],1)
    #pdb.set_trace()
    for a in alpha:
        K = torch.exp(- a * (rx.t() + rx - 2 * xx))
        L = torch.exp(- a * (ry.t() + ry - 2 * yy))
        P = torch.exp(- a * (rx.t() + ry - 2 * zz))
        B = x.size(0)
        beta = (1. / (B * (B - 1)))
        gamma = (2. / (B * B))
        loss += beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)
    return loss
