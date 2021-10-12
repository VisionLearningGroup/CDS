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
from torchvision import transforms
import torch.utils.data as util_data


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


def make_dataset(image_list, class_list):
    class_to_ind = dict(zip(class_list, xrange(len(class_list))))
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            if label in class_list:
                selected_list.append(ind)
                label_list.append(class_to_ind[label])
        image_index = np.array(image_index)
        label_list = np.array(label_list)
        # print(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

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

    def __init__(self, image_list, transform=None, target_transform=None):
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
        return img, target

    def __len__(self):
        return len(self.imgs)


class Imagelists_Office_with_Index(object):
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

    def __init__(self, image_list, transform=None, target_transform=None):
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
        return img, target, index

    def __len__(self):
        return len(self.imgs)



def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)




## for MDD
def make_dataset_mdd(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

class ImageList(object):
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

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset_mdd(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)




class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def load_images(args, batch_size, resize_size=256, mode='train', crop_size=224, is_cen=False, semi=False, is_train=False):


    if args.dataset == 'office_home':
        top = '/research/masaito/OfficeHomeDataset_10072016/split_iccv'
        p_path = os.path.join('/research/masaito/OfficeHomeDataset_10072016/Art')
        class_list = os.listdir(p_path)
        image_set_file_s = os.path.join(top, 'labeled_source_images_' + args.source + '.txt')
        if semi:
            image_set_file_t = os.path.join(top, 'labeled_target_images_' + args.target + '_%d.txt' % (num))
            image_set_file_t_val = os.path.join(top, 'validation_target_images_' + args.target + '_3.txt')
            image_set_file_test = os.path.join(top, 'unlabeled_target_images_' + args.target + '_%d.txt' % (num))
        else:
            image_set_file_t = os.path.join(top, 'labeled_target_images_' + args.target + '_1.txt')
            image_set_file_t_val = os.path.join(top, 'validation_target_images_' + args.target + '_3.txt')
            image_set_file_test = os.path.join(top, 'unlabeled_target_images_' + args.target + '_1.txt')
            # image_set_file_test = os.path.join(top, 'labeled_source_images_' + args.target + '.txt')


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


    if not is_train:
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])

        if mode == 'val':
            images = ImageList(open(image_set_file_t_val).readlines(), transform=transformer)
        elif mode == 'train':
            images = ImageList(open(image_set_file_s).readlines(), transform=transformer)
        elif mode == 'test':
            images = ImageList(open(image_set_file_test).readlines(), transform=transformer)

        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4)

    else:
        if is_cen:
            transformer = transforms.Compose([ResizeImage(resize_size),
                transforms.Scale(resize_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize])
        else:
            transformer = transforms.Compose([ResizeImage(resize_size),
                  transforms.RandomResizedCrop(crop_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize])

        if mode == 'val':
            images = ImageList(open(image_set_file_t_val).readlines(), transform=transformer)
        elif mode == 'train':
            images = ImageList(open(image_set_file_s).readlines(), transform=transformer)
        elif mode == 'test':
            images = ImageList(open(image_set_file_test).readlines(), transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4)

    return images_loader

