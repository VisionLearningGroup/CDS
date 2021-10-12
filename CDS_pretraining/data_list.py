import torch
import numpy as np
import random
import torch.utils.data as data
import os
import os.path
import PIL
from PIL import Image
import pdb
import random
# import cPickle
from torch.utils.data.sampler import WeightedRandomSampler


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def m_hot(labels, num_class):
    vector = np.zeros(num_class)
    for label in labels:
        vector[label] = 1.0
    return vector


def make_dataset_target(image_list, class_list, perclass=5):
    class_to_ind = dict(zip(class_list, xrange(len(class_list))))
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            selected_list.append(ind)
            label_list.append(class_to_ind[label])
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    txs = []
    tys = []
    for j in range(12):
        ind_target = np.where(label_list == j)
        inds = np.random.permutation(ind_target[0].shape[0])
        tys.append(label_list[ind_target[0][inds[:perclass]]])
        txs.append(image_index[ind_target[0][inds[:perclass]]])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)
    return txs, tys
#
#
# def make_dataset(image_list, class_list):
#     class_to_ind = dict(zip(class_list, xrange(len(class_list))))
#     with open(image_list) as f:
#         image_index = [x.split(' ')[0] for x in f.readlines()]
#     with open(image_list) as f:
#         label_list = []
#         selected_list = []
#         for ind, x in enumerate(f.readlines()):
#             label = x.split(' ')[1].strip()
#             if label in class_list:
#                 selected_list.append(ind)
#                 label_list.append(class_to_ind[label])
#         image_index = np.array(image_index)
#         label_list = np.array(label_list)
#         # print(label_list)
#     image_index = image_index[selected_list]
#     return image_index, label_list

def make_dataset_nolist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
        # print(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

class Imagelists_Office(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, transform=None, target_transform=None, mode_self=True):
        #if target:
        #    imgs, labels = make_dataset_target(image_list, class_list)
        #else:
        imgs, labels = make_dataset_nolist(image_list)
        #print(labels)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.mode_self = mode_self
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """


        path = self.imgs[index]
        target = self.labels[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return img, torch.LongTensor(target.astype(np.long)), torch.LongTensor(index)
        if self.mode_self:
            return img, target, index
        else:
            return img, target


    def __len__(self):
        return len(self.imgs)

