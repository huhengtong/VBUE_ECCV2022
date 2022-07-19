# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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
import torchvision.transforms as transforms
import pdb

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.resnet import resnet50, resnet101
from mean_teacher.functions_initial import *
from mean_teacher.resnet18_mini import resnet18
from mean_teacher.WRNet_mini import WideResNet


LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    dirpath = '/cache/10000initial_mini/al_vib2_iter4'
#     dirpath = '/dev/shm/3000labeled_mini_shakeshake/stage2_coreset'
    os.makedirs(dirpath, exist_ok=True)
    #checkpoint_path = os.makedirs(subdir, exist_ok=True)
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, unlabeled_idxs = create_data_loaders(**dataset_config, args=args)
#     train_loader, eval_loader = create_data_loaders_onestage(**dataset_config, args=args)
    #train_loader, eval_loader = create_data_loaders_initial(**dataset_config, args=args)

    def create_model(ema=False):
        #LOG.info("=> creating {ema}model '{arch}'".format(
        #    ema='EMA ' if ema else '',
        #    arch='resnet18'))
        #model = resnet18(num_classes = 100)
        LOG.info("=> creating {ema}model '{arch}'".format(
            ema='EMA ' if ema else '',
            arch='WRN-28-2'))
        model = WideResNet(depth=28, num_classes=100, widen_factor=2, dropRate=0.2)
#         LOG.info("=> creating {ema}model '{arch}'".format(
#             ema='EMA ' if ema else '',
#             arch='resnet50'))
#         model = resnet50(num_classes = 100)
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

    def get_pc(data_idxs, dataset, model):
        model.eval()
        sampler = SubsetRandomSampler(data_idxs)
        batch_sampler = BatchSampler(sampler, 512, drop_last=False)
        pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=args.workers,
                                pin_memory=True)

        pseudo_list = torch.zeros(len(dataset.imgs), 100)
        label_list = torch.zeros(len(dataset.imgs)).long()
        for i, (input, (target, idx)) in enumerate(pred_loader):
            input = input.cuda()
            with torch.no_grad():
                output,_,_ = model(input)
            output_sm = torch.softmax(output.data.cpu(), dim=1)
            pred, label = torch.max(output_sm, dim=1)
            pseudo_list[idx] = output_sm
            label_list[idx] = label
            
        pseudo_list = pseudo_list[data_idxs]
        label_list = label_list[data_idxs]
        return pseudo_list, label_list
    
    def read_mini():
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

        channel_stats = dict(mean=mean_pix,
                         std=std_pix)

        eval_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

        traindir = '/cache/mini_imagenet/train'
        #traindir = args.train_subdir
        dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
        return dataset
    
    dataset = read_mini()  
    data.modify_dataset(dataset)
    pseudo_list = torch.zeros((len(unlabeled_idxs), 100, args.epochs))
    label_epoch = torch.zeros((len(unlabeled_idxs), args.epochs)).long()
    for epoch in range(args.start_epoch, args.epochs):
        print('EPOCH:', epoch)
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch, training_log)
        
        #if epoch >= 80: 
        #    pseudo_list[:,:,epoch], label_epoch[:, epoch] = get_pc(unlabeled_idxs, dataset, ema_model)
        #if epoch == args.epochs - 1:
        #    np.save('/cache/10000initial_mini/al_vib2_iter3/label_epoch_iter3.npy', label_epoch)            
        #    np.save('/cache/10000initial_mini/al_vib2_iter3/pseudo_list_iter3.npy', pseudo_list)
        
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
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, dirpath, epoch + 1)


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
    traindir = '/cache/mini_imagenet/train'
    testdir = '/cache/mini_imagenet/test'

    print(args.labeled_batch_size, args.batch_size)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    labeled_idxs_initial = read_dataset_initial(dataset, 100)
    labeled_idxs_al_iter1 = np.load('/cache/10000initial_mini/al_vib2_iter3/idx_selected_iter1.npy')
    labeled_idxs_al_iter2 = np.load('/cache/10000initial_mini/al_vib2_iter3/idx_selected_iter2.npy')
    labeled_idxs_al_iter3 = np.load('/cache/10000initial_mini/al_vib2_iter3/idx_selected_iter3.npy')
    labeled_idxs_al_iter4 = np.load('/cache/10000initial_mini/al_vib2_iter4/idx_selected_iter4.npy')

    #labeled_idxs = labeled_idxs_initial
    labeled_idxs = np.concatenate((labeled_idxs_initial, labeled_idxs_al_iter1, labeled_idxs_al_iter2, labeled_idxs_al_iter3, labeled_idxs_al_iter4))
    labeled_idxs, unlabeled_idxs = data.relabel_dataset_initial(dataset, labeled_idxs)
#     labeled_idxs, unlabeled_idxs = data.relabel_dataset_bl_stage2(dataset, labeled_idxs, false_pred_dict1, false_pred_dict2)
    print(len(labeled_idxs), len(unlabeled_idxs))

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)       
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
        batch_size=200,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
    return train_loader, test_loader, unlabeled_idxs


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch, log):
    global global_step

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
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
    for i, ((input, ema_input), target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        #meters.update('lr', optimizer.param_groups[0]['lr'])

        #input_var = torch.autograd.Variable(input)
        input_var = input.cuda()
        #ema_input_var = torch.autograd.Variable(ema_input, volatile=True)
        with torch.no_grad():
            ema_input_var = ema_input.cuda()
        #target_var = torch.autograd.Variable(target.cuda(async=True))
        target_var = target.cuda()

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        #meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # ema_model_out = ema_model(ema_input_var)
        # model_out = model(input_var)
        ema_logit, _, _ = ema_model(ema_input_var)
        class_logit, cons_logit, _ = model(input_var)

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)        
#         ema_logit_new = torch.where(false_target.cuda() == 0, torch.tensor(-1000000, dtype=torch.float32).cuda(), ema_logit)

        if args.logit_distance_cost >= 0:
            #class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            #meters.update('res_loss', res_loss.data[0])
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        #print(input.size(), class_logit.size(), target_var.size())
        class_loss = class_criterion(class_logit, target_var) / minibatch_size        
        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            weight = torch.ones(len(ema_logit)).cuda()
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit, weight) / minibatch_size
        else:
            consistency_loss = 0
            
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
#             target_var = target

#         minibatch_size = len(target_var)
#         labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
#         assert labeled_minibatch_size > 0
        #meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        with torch.no_grad():
            output1, _, _ = model(input_var)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        num_correct_iter, num_sample_iter = accuracy(output1.data.cpu(), target.data)
        num_correct += num_correct_iter
        num_all += num_sample_iter

        # measure elapsed time
        #meters.update('batch_time', time.time() - end)
        end = time.time()
    return num_correct / num_all


def save_checkpoint(state, is_best, dirpath, epoch):
    #filename = 'checkpoint.{}.ckpt'.format(epoch)
    filename = 'checkpoint.ckpt'
    #print(dirpath, filename)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    #best_path = os.path.join(best_path, 'best_initial.ckpt')
    torch.save(state, checkpoint_path)
    #LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        #torch.save(state, checkpoint_path)
        shutil.copyfile(checkpoint_path, best_path)
        #LOG.info("--- checkpoint copied to %s ---" % best_path)


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
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target):
    _, prediction = torch.max(output, dim=1)
    num_correct = np.sum(prediction.numpy() == target.numpy())# / len(target)
    return num_correct, len(target)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    import moxing as mox
    if not os.path.exists('/cache/mini_imagenet'):
        src = '.../data/mini_imagenet/'
        dst = '/cache/mini_imagenet'
        mox.file.copy_parallel(src, dst)
    
    main(RunContext(__file__, 0))
    
