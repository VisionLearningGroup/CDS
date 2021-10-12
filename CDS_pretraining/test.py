import torch
import time
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1/total


def kNN_DA(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, verbose=False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainsize =  trainloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()

    trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    C = trainLabels.max() + 1


    with torch.no_grad():

        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4) ## trainloader memory
            for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
                targets = targets.cuda()
                inputs = inputs.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
            trainLabels = torch.LongTensor(temploader.dataset.labels).cuda()
            trainloader.dataset.transform = transform_bak

        lemniscate.memory = trainFeatures.t()




        top1 = 0.
        top5 = 0.
        end = time.time()
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C).cuda()
            for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
                end = time.time()
                inputs = inputs.cuda()
                targets = targets.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                net_time.update(time.time() - end)
                end = time.time()

                dist = torch.mm(features, trainFeatures[:,:trainsize])

                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(targets.data.view(-1, 1))
                cls_time.update(time.time() - end)

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += targets.size(0)

                if verbose:
                    print('Test [{}/{}]\t'
                      'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                      'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                      'Top1: {:.2f}  Top5: {:.2f}'.format(
                    total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

        # logging.info(top1 * 100. / total)

    logging.info('Test [{}/{}]\t'
          'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
          'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
          'Top1: {:.2f}  Top5: {:.2f}'.format(
        total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    return top1 *100./ total



def recompute_memory(epoch, net, lemniscate, trainloader):

    net.eval()
    trainFeatures = lemniscate.memory.t()
    batch_size = 100

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):

            targets = targets.cuda()
            inputs = inputs.cuda()
            c_batch_size = inputs.size(0)
            features = net(inputs)
            features = F.normalize(features)

            trainFeatures[:, batch_idx * batch_size:batch_idx * batch_size + c_batch_size] = features.data.t()


            # if batch_idx * batch_size + c_batch_size > 5000:
            #     break

        trainLabels = torch.LongTensor(temploader.dataset.labels).cuda()
        trainloader.dataset.transform = transform_bak

    lemniscate.memory = trainFeatures.t()

    lemniscate.memory_first = False
