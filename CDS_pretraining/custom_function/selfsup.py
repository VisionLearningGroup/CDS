from __future__ import print_function
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from utils_dh import AverageMeter
from test import NN, kNN_DA, recompute_memory

from return_dataset import return_dataset_selfsup, set_model_self
import logging


def train_selfsup_only(epoch, args, net, name_weight, device, lemniscate_s, lemniscate_t, lemniscate, optimizer, save_load=True, save_weight=True):


    total_time = AverageMeter()

    batch = args.batch_size
    source_loader, target_loader_unl, val_loader, source_loader_val, class_list = return_dataset_selfsup(args, batch_size=batch)

    set_model_self(source_loader, target_loader_unl, val_loader, True)

    cross_entropy_loss = nn.CrossEntropyLoss()

    if lemniscate_s.memory_first:
        recompute_memory(epoch, net, lemniscate_s, source_loader)

    if lemniscate_t.memory_first:
        recompute_memory(epoch, net, lemniscate_t, target_loader_unl)
        acc = kNN_DA(epoch, net, lemniscate, source_loader_val, val_loader, 200, args.nce_t, 1)
        # may get random due to random seed in FC

    net.train()

    start = time.time()

    scaler = torch.cuda.amp.GradScaler()


    for batch_idx, (inputs, targets, indexes) in enumerate(source_loader):

        try:
            inputs2, targets2, indexes2 = target_loader_unl_iter.next()
        except:
            target_loader_unl_iter = iter(target_loader_unl)
            inputs2, targets2, indexes2 = target_loader_unl_iter.next()

        inputs, targets, indexes = inputs.cuda(), targets.cuda(), indexes.type(torch.LongTensor).cuda()
        inputs2, targets2, indexes2 = inputs2.cuda(), targets2.cuda(), indexes2.type(torch.LongTensor).cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            features1 = net(inputs)
            features2 = net(inputs2)

            outputs = lemniscate_s(features1, indexes)
            outputs2 = lemniscate_t(features2, indexes2)

            loss_id = 0

            if args.instance:

                if args.instance:
                    source_cross = (cross_entropy_loss(outputs, indexes))
                    target_cross = cross_entropy_loss(outputs2, indexes2)

                    loss_id = (source_cross + target_cross) / 2.0

                total_loss = loss_id

            ###
            if args.s2t or args.t2s:
                outputs4 = lemniscate_s(features2, indexes2)
                outputs4 = torch.topk(outputs4, min(args.n_neighbor, source_loader.dataset.__len__()), dim=1)[0]
                outputs4 = F.softmax(outputs4*args.temp2, dim=1)
                loss_ent4 = -args.lambda_value * torch.mean(torch.sum(outputs4 * (torch.log(outputs4 + 1e-5)), 1))
                loss_cdm = loss_ent4

                if args.s2t:
                    outputs3 = lemniscate_t(features1, indexes)
                    outputs3 = torch.topk(outputs3,  min(args.n_neighbor, target_loader_unl.dataset.__len__()), dim=1)[0]
                    outputs3 = F.softmax(outputs3*args.temp2, dim=1)
                    loss_ent3 = -args.lambda_value * torch.mean(torch.sum(outputs3 * (torch.log(outputs3 + 1e-5)), 1))
                    loss_cdm += loss_ent3
                    cdm_loss = loss_cdm/2.0

                total_loss += loss_cdm

            else:
                loss_cdm=0

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad()

        lemniscate_s.update_wegiht(features1.detach(), indexes)
        lemniscate_t.update_wegiht(features2.detach(), indexes2)


        if (batch_idx+1) % 200 == 0:
            with torch.no_grad():
                ent1 = F.softmax(outputs)
                ent1 = torch.mean(torch.sum(ent1 * (torch.log(ent1 + 1e-5)), 1))
                ent2 = F.softmax(outputs2)
                ent2 = torch.mean(torch.sum(ent2 * (torch.log(ent2 + 1e-5)), 1))

            logging.info('Epoch: [{}][{}/{}] Loss:{}, ent3:{}'.format(epoch, batch_idx, len(source_loader), loss_id, loss_cdm))
            if args.instance:
                logging.info('Cross Entropy : source:{}  target:{}'.format(source_cross, target_cross))
            logging.info('Entropy : ent_s: {}, ent_t:{}, ent_each:{}'.format(-ent1, -ent2, loss_cdm))

    end = time.time()
    logging.info('time: {}'.format(end-start))

    lemniscate_s.memory_first = False
    lemniscate_t.memory_first = False

    logging.info('source-target')
    acc = kNN_DA(epoch, net, lemniscate, source_loader_val, val_loader, 200, args.nce_t, 1)


    if save_weight:
        state = {
            'net': net.state_dict(),
        }
        torch.save(state,'./checkpoint/{}_epoch_{}.t7'.format(name_weight, epoch))

    return acc

