import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from lib.normalize import Normalize

__all__ = ['AlexNet', 'alexnet', 'alexnet_mme', 'alexnet_dropout', 'alexnet_pure']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}



class AlexNet2(nn.Module):

    def __init__(self, num_classes=1000, low_dim=256):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )
        # self.fc = nn.Linear(4096, low_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # x = self.fc(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, low_dim=256):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),

        )
        # self.fc = nn.Linear(4096, low_dim)
        # self.fc = nn.Linear(4096, low_dim)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # x = self.fc(x)
        x = self.l2norm(x)
        return x


class AlexNet_dropout(nn.Module):

    def __init__(self, num_classes=1000, low_dim=256):
        super(AlexNet_dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),

        )

        # self.fc = nn.Linear(4096, low_dim)
        self.fc = nn.Linear(4096, low_dim)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.fc(x)
        x = self.l2norm(x)
        return x

def alexnet(pretrained=False, progress=True, path=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """


    if path:
        model = AlexNet2(**kwargs)
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

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_state_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    elif pretrained:
        model = AlexNet(**kwargs)
        print('pretrained')
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        # print(state_dict)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        print(sorted(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# def alexnet(pretrained=False, progress=True, path=False, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#
#
#     if path:
#         model = AlexNet2(**kwargs)
#         import torch
#         checkpoint = torch.load(path)
#         pretrained_dict = checkpoint['net']
#         # lemniscate = checkpoint['lemniscate']
#         # best_acc = checkpoint['acc']
#         # start_epoch = checkpoint['epoch']
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#         for k, v in pretrained_dict.items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v
#
#         model_dict = model.state_dict()
#         # 1. filter out unnecessary keys
#         new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(new_state_dict)
#         # 3. load the new state dict
#         model.load_state_dict(model_dict)
#
#     elif pretrained:
#         model = AlexNet(**kwargs)
#         print('pretrained')
#         state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                               progress=progress)
#         # print(state_dict)
#         model_dict = model.state_dict()
#
#         pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
#         print(sorted(pretrained_dict.keys()))
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
#
#     return model


def alexnet_pure(pretrained=False, progress=True, path=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """


    model = AlexNet(**kwargs)
    print('pretrained')
    state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                          progress=progress)
    model.load_state_dict(state_dict)

    return model



def alexnet_dropout(pretrained=False, progress=True, path=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """


    if path:
        model = AlexNet2(**kwargs)
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

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_state_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    elif pretrained:
        model = AlexNet_dropout(**kwargs)
        print('pretrained')
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        # print(state_dict)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        print(sorted(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def alexnet_mme(pretrained=False, progress=True, path=False, **kwargs):



    if path:
        model = AlexNet2(**kwargs)
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

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_state_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)



    elif pretrained:
        model = AlexNet2(**kwargs)
        print('pretrained')
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        # print(state_dict)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        print(sorted(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    return model