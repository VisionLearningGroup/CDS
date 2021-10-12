from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
import pdb
import math
import numpy as np

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        #nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        #nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        #nn.init.zeros_(m.bias)



class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]



class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


def loss_Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def loss_CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduce=False)(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


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
        for i in range(6):
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



class AlexNetBase_selfsup(nn.Module):
    def __init__(self,path=False):
        super(AlexNetBase_selfsup, self).__init__()


        if path:
            checkpoint = torch.load(path)
            pretrained_dict = checkpoint['net']
            # lemniscate = checkpoint['lemniscate']
            # best_acc = checkpoint['acc']
            # start_epoch = checkpoint['epoch']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_alexnet = models.alexnet(pretrained=True)

            model_dict = model_alexnet.state_dict()
            # # 1. filter out unnecessary keys
            new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            # # 2. overwrite entries in the existing state dict
            model_dict.update(new_state_dict)
            # # 3. load the new state dict
            model_alexnet.load_state_dict(model_dict)

        else:
            model_alexnet = models.alexnet(pretrained=True)

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
        for i in range(6):
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


class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        if option =='vgg_bn':
            vgg16=models.vgg11_bn(pretrained=pret)
        else:
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
            x = F.normalize(x)
            x_out = self.fc(x) / self.temp
            return x_out


    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.1,dropout=False):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class)
        self.num_class = num_class
        self.temp = temp
        self.dropout = dropout
    def forward(self, x, reverse=False,eta=0.1):

        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        x_out = self.fc2(x)/0.05
        return x_out



class Predictor_single(nn.Module):
    def __init__(self, num_class=64, low_dim=4096, temp=0.1,dropout=False):
        super(Predictor_single, self).__init__()

        self.fc2 = nn.Linear(low_dim, num_class)
        self.num_class = num_class
        self.temp = temp
        self.dropout = dropout

    def forward(self, x, reverse=False,eta=0.1):

        if reverse:
            x = grad_reverse(x, eta)
        # x = F.normalize(x)
        if self.dropout:
            x = F.dropout(x,training=self.training, p=0.1)
        x_out = self.fc2(x)/0.05
        return x_out



class Predictor_deep_2(nn.Module):
    def __init__(self, num_class=64, inc=4096, low_dim=512,temp=0.1,dropout=False, path=False):
        super(Predictor_deep_2, self).__init__()
        self.fc = nn.Linear(inc, low_dim)
        self.fc2 = nn.Linear(low_dim, num_class)
        self.num_class = num_class
        self.temp = temp
        self.dropout = dropout

        weights_init(self)
        if path:
            import torch
            checkpoint = torch.load(path)
            pretrained_dict = checkpoint['net']
            # lemniscate = checkpoint['lemniscate']
            # best_acc = checkpoint['acc']
            # start_epoch = checkpoint['epoch']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(new_state_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)


    def forward(self, x, reverse=False,eta=0.1):

        x = self.fc(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        x_out = self.fc2(x)/0.05
        return x_out


    def forward_2(self, x, reverse=False,eta=0.1):

        x_feat = self.fc(x)
        if reverse:
            x = grad_reverse(x, eta)
        x_out = F.normalize(x_feat)

        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        x_out = self.fc2(x_out)/0.05
        return x_out, x_feat

class Predictor_deep_id(nn.Module):
    def __init__(self, num_class=64, inc=4096, low_dim=512,temp=0.1,dropout=False, path=False, normalize=False):
        super(Predictor_deep_id, self).__init__()
        self.fc2 = nn.Linear(low_dim, num_class)
        self.num_class = num_class
        self.temp = temp
        self.dropout = dropout
        self.normalize =  normalize
        weights_init(self)

    def forward(self, x, reverse=False,eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        # if self.normalize:
        #     x = F.normalize(x)
        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        x_out = self.fc2(x)/0.05
        return x_out


    def forward_2(self, x, reverse=False,eta=0.1):

        x_feat = self.fc(x)
        if reverse:
            x = grad_reverse(x, eta)
        x_out = F.normalize(x_feat)

        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        x_out = self.fc2(x_out)/0.05
        return x_out, x_feat


class Predictor_deep_id_2(nn.Module):
    def __init__(self, num_class=64, inc=4096, low_dim=512,temp=0.1,dropout=False, path=False, normalize=False):
        super(Predictor_deep_id_2, self).__init__()
        self.fc2 = nn.Linear(low_dim, num_class)
        self.num_class = num_class
        self.temp = temp
        self.dropout = dropout
        self.normalize =  normalize
        weights_init(self)

    def forward(self, x, reverse=False,eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        # if self.normalize:
        #     x = F.normalize(x)
        if self.dropout:
            x = F.dropout(x,training=self.training,p=0.1)
        x_out = self.fc2(x)
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


class Discriminator_2(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator_2, self).__init__()
        self.fc1_1 = nn.Linear(inc, 256)
        self.fc2_1 = nn.Linear(256, 128)
        self.fc3_1 = nn.Linear(128, 1)
    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x,eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out
