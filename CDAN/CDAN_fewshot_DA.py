import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from utils_dh import setup_logging, load_config
import logging


torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False



def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                # labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def image_classification_source(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['source_ul_test'][i]) for i in range(10)]
            for i in range(len(loader['source_ul_test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["source_ul_test"])
            for i in range(len(loader['source_ul_test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                # labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def image_classification_val(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['val'][i]) for i in range(10)]
            for i in range(len(loader['val'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["val"])
            for i in range(len(loader['val'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                # labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy




def train(config, trial=0, args=None):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=data_config["source"]["batch_size"], \
            shuffle=True, num_workers=4, drop_last=True)

    dsets["source_ul"] = ImageList(open(data_config["source_ul"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source_ul"] = DataLoader(dsets["source_ul"], batch_size=data_config["source_ul"]["batch_size"], \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["source_ul_test"] = ImageList(open(data_config["source_ul_test"]["list_path"]).readlines(), \
                                   transform=prep_dict["test"])
    dset_loaders["source_ul_test"] = DataLoader(dsets["source_ul_test"], batch_size=data_config["source_ul_test"]["batch_size"], \
                                           shuffle=False, num_workers=4)


    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["val"] = ImageList(open(data_config["val"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["val"] = DataLoader(dsets["val"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])


    checkpoint_path = '../CDS_pretraining/checkpoint/{}_epoch_{}.t7'.format(args.name_weight, trial)
    checkpoint = torch.load(checkpoint_path)
    checkpoint = checkpoint['net']

    # checkpoint.pop('fc.weight')
    # checkpoint.pop('fc.bias')
    base_network.load_state_dict(checkpoint, strict=False)
    logging.info('load weight')


    base_network = base_network.cuda()


    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        if args.method == 'DANN':
            ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
        else:
            ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()




    if args.freeze:
        parameter_list = base_network.get_fc_layers() + ad_net.get_parameters()

    else:
        parameter_list = base_network.get_parameters() + ad_net.get_parameters()



    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        
    counter = 0
    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_source_ul = len(dset_loaders["source_ul"])
    len_train_target = len(dset_loaders["target"])

    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0

    best_acc = 0.0

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)


            temp_model = nn.Sequential(base_network)
            temp_acc = image_classification_test(dset_loaders, \
                                                 base_network, test_10crop=prep_config["test_10crop"])
            temp_source_acc = image_classification_source(dset_loaders, \
                                                          base_network, test_10crop=prep_config["test_10crop"])


            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            # print(log_str)
            print(args.name)
            logging.info("iter: {:05d}, c_acc: {:.4f} |  c_source: {:.4f} ".format(i, temp_acc, temp_source_acc))


        loss_params = config["loss"]                  
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_source_ul == 0:
            iter_source_ul = iter(dset_loaders["source_ul"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_source_ul, labels_source_ul = iter_source_ul.next()
        inputs_target, labels_target = iter_target.next()

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        inputs_source_ul, labels_source_ul = inputs_source_ul.cuda(), labels_source_ul.cuda()

        features_source, outputs_source = base_network(inputs_source)

        if config['method'] != 'S':
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer, batch_size=args.l_batch_size)
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net, batch_size=args.l_batch_size)
        elif config['method'] == 'S':
            transfer_loss = 0
        else:
            raise ValueError('Method cannot be recognized.')


        if args.ul_method == 'label':
            ul_loss = 0

        elif args.ul_method == 'ENT':
            features_source_ul, outputs_source_ul = base_network(inputs_source_ul)
            source_entropy = torch.mean(loss.Entropy(nn.Softmax(dim=1)(outputs_source_ul))) * args.ent_weight
            ul_loss = source_entropy


        classifier_loss = nn.CrossEntropyLoss(ignore_index=-1)(outputs_source, labels_source)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + ul_loss
        # total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()


    # torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN+E', 'DANN', 'S'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])

    parser.add_argument('--s_dset_path', type=str, default='../../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=250, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")


    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='M', help='batch_size')
    parser.add_argument('--n_neighbor', default=700, type=int,
                        metavar='M', help='momentum for non-parametric updates')
    parser.add_argument('--temp2', default=1.0, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--lambda_value', default=1.0, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='validation phase')
    parser.add_argument('--training_da', action='store_false', default=True,
                        help='validation phase')
    parser.add_argument('--source', type=str, default='Real', metavar='B',
                        help='board dir')
    parser.add_argument('--target', type=str, default='Clipart', metavar='B',
                        help='board dir')
    parser.add_argument('--net2', type=str, default='resnet50', metavar='B',
                        help='which network ')
    parser.add_argument('--dataset', type=str, default='office_home',
                        choices=['office', 'office_home', 'cub'],
                        help='the name of dataset, multi is large scale dataset')
    parser.add_argument('--DC', action='store_true', default=False,
                        help='validation phase')
    parser.add_argument('--instance', action='store_true', default=True,
                        help='validation phase')
    parser.add_argument('--t2s', action='store_true', default=True,
                        help='validation phase')
    parser.add_argument('--s2t', action='store_true', default=True,
                        help='validation phase')

    parser.add_argument('--early', action='store_false', default=False,
                        help='early stopping on validation or not')
    parser.add_argument('--patience', default=10, type=int,
                        metavar='M', help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='M', help='batch_size')

    parser.add_argument('--eps', default=10, type=float,
                        metavar='M', help='batch_size')
    parser.add_argument('--ent_weight', default=0.01, type=float,
                        metavar='M', help='batch_size')

    parser.add_argument('--ul_method', type=str, default='ENT', choices=['label', 'ENT'])

    parser.add_argument('--l_batch_size', default=34, type=int,
                        metavar='M', help='batch_size')
    parser.add_argument('--ul_batch_size', default=34, type=int,
                        metavar='M', help='batch_size')

    parser.add_argument('--py_seed', default=0, type=int,
                        metavar='M', help='batch_size')
    parser.add_argument('--fixed_iter', default=3001, type=int,
                        metavar='M', help='batch_size')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='validation phase')

    args = parser.parse_args()

    seed = args.py_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if args.dataset=='office_home':

        shots = [1, 3]
        args.test_interval = 250
        checkpoint = 6
        args.ent_weight = 0.01


    args.name = '{}_{}_{}_{}_{}_to_{}_method'.format(args.dataset, args.method, args.ul_method, args.method, args.source,args.target, args.method)
    setup_logging(args.name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    for shot in shots:

        seed_acc_cds = []

        for seed in [1,2,3]:
            args.py_seed= seed

            args.shot = shot
            logging.info("shot: {:.2f}".format(args.shot))

            load_config(args)

            logging.info(args.s_dset_path)
            logging.info(args.s_dset_path_ul)
            logging.info(args.name_weight)

            # train config
            config = {}
            config['method'] = args.method
            config["gpu"] = args.gpu_id
            config["num_iterations"] = args.fixed_iter
            config["test_interval"] = args.test_interval
            config["snapshot_interval"] = args.snapshot_interval
            config["output_for_test"] = True
            config["output_path"] = "./snapshot/" + args.output_dir
            if not osp.exists(config["output_path"]):
                os.system('mkdir -p '+config["output_path"])
            config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
            if not osp.exists(config["output_path"]):
                os.mkdir(config["output_path"])

            config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
            config["loss"] = {"trade_off":1.0}
            if "AlexNet" in args.net:
                config["prep"]['params']['alexnet'] = True
                config["prep"]['params']['crop_size'] = 227
                config["network"] = {"name":network.AlexNetFc, \
                    "params":{"use_bottleneck":True, "bottleneck_dim":512, "new_cls":True} }
            elif "ResNet" in args.net:
                config["network"] = {"name":network.ResNetFc, \
                    "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":512, "new_cls":True} }
            elif "VGG" in args.net:
                config["network"] = {"name":network.VGGFc, \
                    "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":512, "new_cls":True} }
            config["loss"]["random"] = args.random
            config["loss"]["random_dim"] = 1024

            config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                                   "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                                   "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

            config["dataset"] = args.dataset

            config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.l_batch_size}, \
                              "source_ul": {"list_path": args.s_dset_path_ul, "batch_size": args.ul_batch_size}, \
                              "source_ul_test": {"list_path": args.s_dset_path_ul, "batch_size": 36}, \
                              "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                              "val": {"list_path": args.val_dset_path, "batch_size": 36}, \
                              "test":{"list_path":args.t_dset_path, "batch_size":36}}

            if config["dataset"] == "office":
                if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                   ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                   ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                   ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
                    config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                     ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
                    config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
                config["network"]["params"]["class_num"] = 31
            elif config["dataset"] == "image-clef":
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                config["network"]["params"]["class_num"] = 12
            elif config["dataset"] == "visda":
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                config["network"]["params"]["class_num"] = 12
                config['loss']["trade_off"] = 1.0
            elif config["dataset"] == "office_home":
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                config["network"]["params"]["class_num"] = 65
            elif config["dataset"] == "domainnet":
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                config["network"]["params"]["class_num"] = 126
            elif config["dataset"] == "cub":
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                config["network"]["params"]["class_num"] = 200
            elif config["dataset"] == "cars":
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
                config["network"]["params"]["class_num"] = 281
            else:
                raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
            config["out_file"].write(str(config))
            config["out_file"].flush()


            acc = train(config, checkpoint, args)
            seed_acc_cds.append(acc)

        logging.info('Shot:{:.2f} - three runs averaged - CDS acc: {} +- {}'
                     .format(args.ratio, np.array(seed_acc_cds).mean(),np.array(seed_acc_cds).std()))