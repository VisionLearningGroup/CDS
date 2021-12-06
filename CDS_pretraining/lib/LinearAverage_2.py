import torch
from torch.autograd import Function
from torch import nn
import math
import torch.nn.functional as F

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]));
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory =  self.memory.cuda()
        self.memory_first = True

    def forward(self, x, y):
        out = torch.mm(x, self.memory.t())/self.T
        return out

    def update_wegiht(self, features, index):

        weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)#.cuda()


    def set_weight(self, features, index):

        self.memory.index_select(0, index.data.view(-1)).resize_as_(features)