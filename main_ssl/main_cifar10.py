import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import pdb

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.functions_initial import *
from mean_teacher.resnet_cifar10 import ResNet18
from mean_teacher.WRNet import WideResNet

import pickle
import sys
from torchvision.datasets import CIFAR100, CIFAR10
import matplotlib.image
import scipy.fftpack as fftpack

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    dirpath = '/dev/shm/250initial_cifar10'
    os.makedirs(dirpath, exist_ok=True)
    #checkpoint_path = os.makedirs(subdir, exist_ok=True)
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, labeled_idxs, unlabeled_idxs = create_data_loaders(**dataset_config, args=args)
    #train_loader, eval_loader = create_data_loaders_onestage(**dataset_config, args=args)
#     train_loader, eval_loader = create_data_loaders_initial(**dataset_config, args=args)

    def create_model(ema=False):        
#         LOG.info("=> creating {ema}model '{arch}'".format(
#             ema='EMA ' if ema else '',
#             arch='resnet18'))
#         model = ResNet18()
        LOG.info("=> creating {ema}model '{arch}'".format(
            ema='EMA ' if ema else '',
            arch='WRN-28-2'))
        model = WideResNet(depth=28, num_classes=10, widen_factor=2)
        model = nn.DataParallel(model).cuda()
        
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return
    
    def compute_cLoss(data_idxs, dataset, ema_model):
        model.eval()
        ema_model.eval()
        sampler = SubsetRandomSampler(data_idxs)
        batch_sampler = BatchSampler(sampler, 512, drop_last=False)
        pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=args.workers,
                                pin_memory=True)
        closs_list = torch.zeros(len(dataset.imgs))
#         pseudo_list = torch.zeros(len(dataset.imgs), 10)
#         label_list = torch.zeros(len(dataset.imgs)).long()
        for i, ((input, ema_input), (target, idx)) in enumerate(pred_loader):
            input, ema_input = input.cuda(), ema_input.cuda()
            with torch.no_grad():
                output,_,_ = ema_model(input)
                output_ema,_,_ = ema_model(ema_input)
                
            output_sm = torch.softmax(output.data.cpu(), dim=1)
            output_ema_sm = torch.softmax(output_ema.data.cpu(), dim=1)
#             pred, label = torch.max(output_ema_sm, dim=1)
#             c_loss = F.mse_loss(output_sm, output_ema_sm, reduction='none')
            c_loss = F.pairwise_distance(output_sm, output_ema_sm, p=2)
#             pseudo_list[idx] = output_ema_sm
#             label_list[idx] = label
            closs_list[idx] = c_loss
        closs_list = closs_list[data_idxs]
#         pseudo_list = pseudo_list[data_idxs]
#         label_list = label_list[data_idxs]
        return closs_list#, pseudo_list, label_list
    
    def get_pc(data_idxs, dataset, model):
        model.eval()
        sampler = SubsetRandomSampler(data_idxs)
        batch_sampler = BatchSampler(sampler, 512, drop_last=False)
        pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=args.workers,
                                pin_memory=True)
#         pred_list = torch.zeros(len(dataset.imgs))
        pseudo_list = torch.zeros(len(dataset.imgs), 10)
        label_list = torch.zeros(len(dataset.imgs)).long() +100
        for i, (input, (target, idx)) in enumerate(pred_loader):
            input = input.cuda()
            with torch.no_grad():
                output,_,_ = model(input)
            output_sm = torch.softmax(output.data.cpu(), dim=1)
            pred, label = torch.max(output_sm, dim=1)
#             pred_list[idx] = pred
            pseudo_list[idx] = output_sm
            label_list[idx] = label
            
#         pred_list = pred_list[data_idxs]
        pseudo_list = pseudo_list[data_idxs]
        label_list = label_list[data_idxs]
        return pseudo_list, label_list   
    
    def prob_vibration(pseudo_list, weight, start, end):
        array_size = pseudo_list[:, start:end].shape[1]
        yf_fft = fftpack.fft(pseudo_list[:, start:end], axis=1)
    #     yf_fft_quadsum = np.sum(np.power(abs(yf_fft), 2), axis=1).reshape(-1, 1)
    #     yf = np.power(abs(yf_fft), 2) / np.tile(yf_fft_quadsum, array_size)
    #     yf_half = yf[:, range(int(array_size / 2))] * 2
        yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
        vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
        return vibration_measure
    
    def flip_vibration(cLoss_list, weight, start, end):
        array_size = cLoss_list[:, start:end].shape[1]
        yf_fft = fftpack.fft(cLoss_list[:, start:end], axis=1)
        yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
        vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
        return vibration_measure  
    
    def get_vibration(pseudo_list_all, time_label_list, unlabeled_idxs, best_epoch): 
        #def entropy(prob, base=None):
        #    return np.sum(-np.log(prob)*prob,axis=1)
        def normlize(input_list):
            return (input_list-min(input_list))/ (max(input_list)-min(input_list))
#         def get_rankInfo(input_list):
#             list_rank = np.zeros(len(input_list))
#             list_rank[np.argsort(input_list)] = range(len(input_list))
#             return list_rank
        
        pseudo_list = np.asarray(pseudo_list_all[range(len(unlabeled_idxs)), time_label_list[:, best_epoch], :])
        weight_conf = 0.01
        vibration_conf = prob_vibration(pseudo_list, weight_conf, max(best_epoch-50, 51), best_epoch)

        ### prediction flip vibration (sharpen the curve)
        pred_flip = np.zeros(time_label_list.shape)
        for i in range(len(time_label_list)):
            pred_flip[i, 0:] = time_label_list[i, 0:] == time_label_list[i, best_epoch]
        weight_flip = 0.01
        vibration_flip = flip_vibration(pred_flip, weight_flip, max(best_epoch-50, 51), best_epoch)

        vibration_fused = 0.8*normlize(vibration_conf) + 0.2*normlize(vibration_flip) #+ 0.01*normlize(vibration_ent)
        return vibration_fused
    
    def run_training(dataset_eval, unlabeled_idxs, weights_total, labels_training, weights_class):        
        pseudo_list = torch.zeros((len(unlabeled_idxs), 10, args.epochs))
        label_epoch = torch.zeros((len(unlabeled_idxs), args.epochs)).long()

        best_prec1 = 0
        for epoch in range(args.start_epoch, args.epochs):
            print('EPOCH:', epoch)
            start_time = time.time()
            # train for one epoch        
            train(train_loader, model, ema_model, optimizer, epoch, training_log, weights_total, labels_training, weights_class)
            
            if epoch >= 50: 
                pseudo_list[:,:,epoch], label_epoch[:, epoch] = get_pc(unlabeled_idxs, dataset_eval, ema_model)   
            if epoch == args.epochs - 1:
                np.save('/dev/shm/250initial_cifar10/label_epoch_iter.npy', label_epoch)         
                np.save('/dev/shm/250initial_cifar10/pseudo_list_iter.npy', pseudo_list)
    #             np.save('/dev/shm/3000initial_cifar10/pc_AL_iter1/cLoss_epoch_iter1.npy', cLoss_epoch) 

            #if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1)
            print('accuracy', prec1)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1)
            print('ema_accuracy', ema_prec1)
            #LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
            print('current best accuracy:', best_prec1)

            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, dirpath, epoch)
            
    def get_weights(dataset_eval, labeled_idxs, unlabeled_idxs, best_epoch, iter):
        pseudo_labels_iter0 = np.load('/dev/shm/250initial_cifar10/pseudo_list_iter.npy')
        labels_epoch_iter0 = np.load('/dev/shm/250initial_cifar10/label_epoch_iter.npy')
        print(len(unlabeled_idxs), labels_epoch_iter0.shape)
        
        ### compute the weights for all samples
        weights_total = torch.ones(len(dataset_eval.imgs))   
        vibration_iter0 = get_vibration(pseudo_labels_iter0, labels_epoch_iter0, unlabeled_idxs, best_epoch)
        weights_iter0 = 1 - (vibration_iter0 - min(vibration_iter0)) / (max(vibration_iter0) - min(vibration_iter0))
        weights_total[unlabeled_idxs] = torch.from_numpy(weights_iter0).float()
        
        ### select the easy pseudo labels
        if iter >= 8:
            increase = 0
        else:
            increase = 1000
        inds_easy = np.argsort(vibration_iter0)[0:(28000+increase*iter)]
        labels_training = np.zeros((len(dataset_eval.imgs)), dtype=np.int32) - 2
        labels_training[unlabeled_idxs[inds_easy]] = labels_epoch_iter0[inds_easy, best_epoch]
        
        ### compute the weights for each class
        num_sample_each_class = torch.zeros(10)
        for idx in labeled_idxs:
            path, (label_idx, idx) = dataset_eval.imgs[idx]
            num_sample_each_class[label_idx] += 1
        for i in range(10):
            num_sample_each_class[i] += np.sum(labels_training == i)
        weights_class = (len(inds_easy)+len(labeled_idxs))/10 / num_sample_each_class
        weights_class = weights_class / max(weights_class)
        labels_training = torch.from_numpy(labels_training).long()
        
        return weights_total, labels_training, weights_class
        
    _, dataset_eval = data.read_cifar10()  
    data.modify_dataset(dataset_eval)
#     best_epoch = 126

    ### initialize
    weights_total = torch.ones(len(dataset_eval.imgs)).float()
    labels_training = torch.zeros(len(dataset_eval.imgs)).long() - 2
    weights_class = torch.ones(10)
    
    for iter in range(20):
        print('start training iter:', iter)
        run_training(dataset_eval, unlabeled_idxs, weights_total, labels_training, weights_class)
        ckpt_path = '/dev/shm/250initial_cifar10/best.ckpt'
        checkpoint = torch.load(ckpt_path)
        best_epoch = checkpoint['epoch']
        if best_epoch <= 50:
            best_epoch = 100
        weights_total, labels_training, weights_class = get_weights(dataset_eval, labeled_idxs, unlabeled_idxs, best_epoch, iter)
        

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        #datadir,
                        args):
    traindir = '/dev/shm/cifar10_train'
    testdir = '/dev/shm/cifar10_test'

    print(args.labeled_batch_size, args.batch_size)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    labeled_idxs_initial = read_dataset_initial(dataset, 25)
#     rand_select = np.random.choice(sorted(set(range(len(dataset.imgs))) - set(labeled_idxs_initial)), 1000, replace=False)
#     labeled_idxs_al_iter1 = np.load('/dev/shm/100initial_cifar10/final_vibration_iter1/idx_selected_iter1.npy')
#     labeled_idxs_al_iter2 = np.load('/dev/shm/3000initial_cifar10/fused_fft_AL_iter3/idx_selected_iter2.npy')
#     labeled_idxs_al_iter3 = np.load('/dev/shm/3000initial_cifar10/fused_fft_AL_iter3/idx_selected_iter3.npy')
#     labeled_idxs_al_iter4 = np.load('/dev/shm/3000initial_cifar10/fused_fft_AL_iter4/idx_selected_iter4.npy')
    
#     plabels_easy = np.load('/dev/shm/3000initial_cifar10/final_vibration_iter1/pLabels_easy.npy')
    labeled_idxs = labeled_idxs_initial
#     labeled_idxs = np.concatenate((labeled_idxs_initial, rand_select))  
#     labeled_idxs = np.concatenate((labeled_idxs_initial, labeled_idxs_al_iter1))#, labeled_idxs_al_iter2, labeled_idxs_al_iter3, labeled_idxs_al_iter4))
    labeled_idxs, unlabeled_idxs = data.relabel_dataset_idx(dataset, labeled_idxs)
    print(len(labeled_idxs), len(unlabeled_idxs))

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        # sampler = SubsetRandomSampler(range(len(dataset)))
        # dataset.labeled_idx = range(len(dataset))
        # batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                   batch_sampler=batch_sampler,
                                   num_workers=args.workers,
                                   pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testdir, eval_transformation),
        batch_size=400,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, test_loader, labeled_idxs, np.asarray(unlabeled_idxs)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch, log, weights, plabels_easy, weights_class):
    global global_step
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL, weight=weights_class).cuda()
#     PartSumCELoss = losses.PartSumCrossEntropyLoss().cuda()
#     PartMeanCELoss = losses.PartMeanCrossEntropyLoss().cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    end = time.time()  

#     for i, ((input, ema_input), (target, false_target)) in enumerate(train_loader): 
    for i, ((input, ema_input), (target, img_id)) in enumerate(train_loader): 
#     for i, ((input, ema_input), target) in enumerate(train_loader):        
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        
        input_var = input.cuda()
        weight = weights[img_id].cuda()
        labels_easy = plabels_easy[img_id]
#         neg_labels = neg_labels_all[img_id]
        with torch.no_grad():
            ema_input_var = ema_input.cuda()
        #target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target)
        labeled_minibatch_size = target.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        ema_logit, _,_ = ema_model(ema_input_var)
        class_logit, cons_logit,_ = model(input_var)
        
        # inject easy pseudo labels
        inds = (labels_easy > -2).nonzero()
        if len(inds) > 0:
            target[inds] = labels_easy[inds]         
        target_var = target.cuda()
        
        ### negative labels utilization
        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)       
#         ema_logit_new = ema_logit.data.cpu()
#         neg_label_idx1, neg_label_idx2 = np.where(neg_labels == 1)
#         ema_logit_new[neg_label_idx1, neg_label_idx2] = -1000000
#         ema_logit_new = ema_logit_new.cuda()

        ### positive labels utilization
#         ema_logit_new = ema_logit.data.cpu()
#         inds = (labels_easy > -2).nonzero()
#         ema_logit_new[inds, labels_easy[inds]] = 2*torch.max(ema_logit_new[i])
#         ema_logit_new = ema_logit_new.cuda()

        if args.logit_distance_cost >= 0:
            #class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        class_loss = class_criterion(class_logit, target_var) / minibatch_size
#         class_loss = torch.sum(torch.mul(class_loss, weights_labeled))
#         class_loss = PartMeanCELoss(class_logit, targets_multi.cuda())
        
        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            #meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit, weight) / minibatch_size
            #meters.update('cons_loss', consistency_loss.data[0])
        else:
            consistency_loss = 0
            #meters.update('cons_loss', 0)
        #print(class_loss.item(), consistency_loss.item(), res_loss.item())
        loss = class_loss + consistency_loss + res_loss
        if loss.item() > 1e5:
            print(class_loss.item(), consistency_loss.item(), res_loss.item())
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)


def validate(eval_loader, model, log, global_step, epoch):
    #class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    iter_num, prec = 0, 0
    num_all, num_correct = 0, 0
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        with torch.no_grad():
            input_var = input.cuda()
            target_var = target

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        #meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        with torch.no_grad():
            output1, _,_ = model(input_var)
        # print(output1)
        # pdb.set_trace()
        #softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
        #class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        num_correct_iter, num_sample_iter = accuracy(output1.data.cpu(), target_var.data)
        num_correct += num_correct_iter
        num_all += num_sample_iter

        # measure elapsed time
        #meters.update('batch_time', time.time() - end)
        end = time.time()

    return num_correct / num_all


def save_checkpoint(state, is_best, dirpath, epoch):
#     filename = 'checkpoint.{}.ckpt'.format(epoch)
    filename = 'checkpoint.ckpt'
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    #LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        #torch.save(state, checkpoint_path)
        shutil.copyfile(checkpoint_path, best_path)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     if epoch <= args.consistency_rampup:
#         return args.consistency/10 * ramps.sigmoid_rampup(epoch, 5)
#     else:
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target):
    _, prediction = torch.max(output, dim=1)
    # print(prediction)
    # pdb.set_trace()
    num_correct = np.sum(prediction.numpy() == target.numpy()) #/ len(target)
    return num_correct, len(target)

def unpack_cifar10():
    work_dir = '/dev/shm/'
    test_dir = '/dev/shm/cifar10_test'
    train_dir = '/dev/shm/cifar10_train'

    cifar10 = CIFAR10(work_dir, download=False)

    def load_file(file_name):
        with open(os.path.join(work_dir, cifar10.base_folder, file_name), 'rb') as meta_f:
            return pickle.load(meta_f, encoding="latin1")


    def unpack_data_file(source_file_name, target_dir, start_idx):
        print("Unpacking {} to {}".format(source_file_name, target_dir))
        data = load_file(source_file_name)
        for idx, (image_data, label_idx) in enumerate(zip(data['data'], data['labels'])):
            subdir = os.path.join(target_dir, label_names[label_idx])
            name = "{}_{}.png".format(start_idx + idx, label_names[label_idx])
            os.makedirs(subdir, exist_ok=True)
            image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
            matplotlib.image.imsave(os.path.join(subdir, name), image)
        return len(data['data'])


    label_names = load_file('batches.meta')['label_names']
    print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))

    start_idx = 0
    for source_file_path, _ in cifar10.test_list:
        start_idx += unpack_data_file(source_file_path, test_dir, start_idx)

    start_idx = 0
    for source_file_path, _ in cifar10.train_list:
        start_idx += unpack_data_file(source_file_path, train_dir, start_idx)

if __name__ == '__main__':
    import moxing as mox
    if not os.path.exists('/dev/shm/cifar10_train'):
        src = '.../data/cifar-10-batches-py/'
        dst = '/dev/shm/cifar-10-batches-py'
        mox.file.copy_parallel(src, dst)
        unpack_cifar10()    

    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))

