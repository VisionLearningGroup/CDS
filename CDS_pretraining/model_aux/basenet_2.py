from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
import pdb
import math

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)



class GradMulti(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)


def grad_multi(x, lambd=1.0):
    return GradMulti(lambd)(x)
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


def conv3x3(in_planes, out_planes, stride=1, sn=False):
    "3x3 convolution with padding"
    if not sn:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=True)
    else:
        return SNConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution no padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt() + self.eps
        x /= norm.expand_as(x)
        out = self.weight.unsqueeze(0).expand_as(x) * x
        return out


class AlexNetBase(nn.Module):
    def __init__(self,pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        #print(model_alexnet.features)
        self.conv1 = model_alexnet.features[0]
        self.relu1 = model_alexnet.features[1]
        self.pool1 = model_alexnet.features[2]
        self.conv2 = model_alexnet.features[3]
        self.relu2 = model_alexnet.features[4]
        self.pool2 = model_alexnet.features[5]
        self.conv3 = model_alexnet.features[6]
        self.relu3 = model_alexnet.features[7]
        self.conv4 = model_alexnet.features[8]
        self.relu4 = model_alexnet.features[9]
        self.conv5 = model_alexnet.features[10]
        self.relu5 = model_alexnet.features[11]
        self.pool3 = model_alexnet.features[12]

        self.feature1 = nn.Sequential(*list(model_alexnet.features._modules.values())[:6])
        self.feature2 = nn.Sequential(*list(model_alexnet.features._modules.values())[6:])

        #self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in xrange(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x, target=False, lamda=0.1):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        if target:
            x = grad_multi(x,lamda)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv3(x)
        x = self.relu3(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv4(x)
        x = self.relu4(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        #x = self.feature1(x)
        #x = self.feature2(x)
        feature = x
        x = x.view(x.size(0), 256 * 6 * 6)
        #x = F.normalize(x)
        x = self.classifier(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x

    def output_num(self):
        return self.__in_features


class AlexNetBase_2(nn.Module):
    def __init__(self, pret=True, inc=4096):
        super(AlexNetBase_2, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        # print(model_alexnet.features)
        self.conv1 = model_alexnet.features[0]
        self.relu1 = model_alexnet.features[1]
        self.pool1 = model_alexnet.features[2]
        self.conv2 = model_alexnet.features[3]
        self.relu2 = model_alexnet.features[4]
        self.pool2 = model_alexnet.features[5]
        self.conv3 = model_alexnet.features[6]
        self.relu3 = model_alexnet.features[7]
        self.conv4 = model_alexnet.features[8]
        self.relu4 = model_alexnet.features[9]
        self.conv5 = model_alexnet.features[10]
        self.relu5 = model_alexnet.features[11]
        self.pool3 = model_alexnet.features[12]

        self.feature1 = nn.Sequential(*list(model_alexnet.features._modules.values())[:6])
        self.feature2 = nn.Sequential(*list(model_alexnet.features._modules.values())[6:])

        # self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in xrange(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

        self.classifier.classifier4 = nn.Linear(4096, inc)

    def forward(self, x, target=False, lamda=0.1):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv3(x)
        x = self.relu3(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv4(x)
        x = self.relu4(x)
        if target:
            x = grad_multi(x, lamda)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        # x = self.feature1(x)
        # x = self.feature2(x)
        feature = x
        x = x.view(x.size(0), 256 * 6 * 6)
        # x = F.normalize(x)
        x = self.classifier(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, option='resnet18', pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x, source=True,target=False):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)

        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.1,
                 cosine=False,dropout=False):
        super(Predictor, self).__init__()
        if cosine:
            self.fc = nn.Linear(inc, num_class, bias=False)
        else:
            self.fc = nn.Linear(inc, num_class,bias=True)
        self.num_class = num_class
        self.temp = temp
        self.cosine = cosine
        self.dropout = dropout




    def forward(self, x, reverse=False,eta=0.1):
        if reverse:
            x = grad_reverse(x,eta)
        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        if not self.cosine:
            x_out = self.fc(x)
            return x_out
        else:
            x = F.normalize(x) * 10
            x_out = self.fc(x) / self.temp
            return x_out


import numpy as np

class Predictor2(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.1,
                 cosine=False, dropout=False):
        super(Predictor2, self).__init__()

        self.num_class = num_class
        self.final_dim = 2000
        self.scale = 2.0

        A = np.random.rand(self.final_dim, self.num_class) * 1.0

        self.v, R = np.linalg.qr(A)

        for i in range(self.num_class):
            self.v[:, i] = (self.v[:, i] / np.linalg.norm(self.v[:, i])) * 10.0
            # self.v[:, i] = (self.v[:, i] / np.linalg.norm(self.v[:, i]))

        self.v = torch.from_numpy(self.v).float()
        self.v_cuda = self.v.cuda()

        self.fc = nn.Linear(inc, self.final_dim)
        self.temp=temp

    def forward(self, x):

        x_out = self.fc(x)
        x_out = F.normalize(x_out) * 10.0
        # x_out = self.fc(x) / self.temp
        # return x_out
        x_out = torch.mm(x_out, self.v_cuda)
        return x_out


    def forward2(self, x):
        x_out = self.fc(x)
        x_out = torch.bmm(x_out, self.v_cuda)
        return x_out


class Predictor_2(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.1,
                 cosine=False,dropout=False):
        super(Predictor_2, self).__init__()

        if cosine:
            self.fc =  torch.nn.utils.weight_norm(nn.Linear(inc, num_class, bias=False), name='weight')
        else:
            self.fc = torch.nn.utils.weight_norm(nn.Linear(inc, num_class, bias=True), name='weight')

        # if cosine:
        #     self.fc = nn.Linear(inc, num_class, bias=False)
        # else:
        #     self.fc = nn.Linear(inc, num_class,bias=True)

        self.num_class = num_class
        self.temp = temp
        self.cosine = cosine
        self.dropout = dropout
    def forward(self, x, reverse=False,eta=0.1):
        if reverse:
            x = grad_reverse(x,eta)
        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)

        # with torch.no_grad():
        #     self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))

        if not self.cosine:
            x_out = self.fc(x)
            return x_out
        else:
            x = F.normalize(x)
            x_out = self.fc(x) / self.temp
            return x_out


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)
    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x,eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out
