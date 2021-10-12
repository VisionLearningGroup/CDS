from torchvision import datasets, transforms
import os
import torch
from data_list import  Imagelists_Office
import logging
import numpy as np
from PIL import ImageFilter


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))



def return_dataset_selfsup(args, num=3,semi=False,uda=False, shuffle=True, batch_size=36, bs1=2,bs2=2):

    if args.dataset == 'office_home':
        class_list = range(65)
        top = '../data/office_home/'
        image_set_file_s = os.path.join(top, 'labeled_source_images_' + args.source + '.txt')
        image_set_file_test = os.path.join(top, 'labeled_source_images_' + args.target + '.txt')


    crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    source_dataset = Imagelists_Office(image_set_file_s, transform=data_transforms['train'])
    target_dataset = Imagelists_Office(image_set_file_test, transform=data_transforms['val'])
    target_dataset_val = Imagelists_Office(image_set_file_test, transform=data_transforms['test'])
    source_dataset_val = Imagelists_Office(image_set_file_s, transform=data_transforms['test'])

    if args.net == 'alexnet':
        bs = 128
    else:
        bs = batch_size

    if args.dataset == 'visda':
        source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs*args.bs1, num_workers=4, shuffle=True, drop_last=True)
    else:
        source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs*args.bs1, num_workers=4, shuffle=True,drop_last=True)

    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=bs*args.bs2, num_workers=4, shuffle=True, drop_last=True)
    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=bs*args.bs2, num_workers=6, shuffle=False, drop_last=False)
    source_loader_val = torch.utils.data.DataLoader(source_dataset_val, batch_size=bs, num_workers=6, shuffle=False,
                                                    drop_last=False)

    return source_loader, target_loader, target_loader_val, source_loader_val, class_list



def set_model_self(source_loader, target_loader, target_loader_val, model_self, target_loader_test=None, source_loader_test=None):
    source_loader.mode_self = model_self
    target_loader.mode_self = model_self
    target_loader_val.mode_self = model_self

    if target_loader_test:
        target_loader_test.mode_self = model_self
        source_loader_test.mode_self = model_self

