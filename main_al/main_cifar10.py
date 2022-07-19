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
import torchvision.transforms as transforms

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

    dirpath = '/cache/100initial_cifar10/al_vib2_iter1'
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
        #LOG.info("=> creating {ema}model '{arch}'".format(
        #    ema='EMA ' if ema else '',
        #    arch='resnet18'))
        #model = ResNet18(num_classes=10)
        #model = nn.DataParallel(model).cuda()
        LOG.info("=> creating {ema}model '{arch}'".format(
            ema='EMA ' if ema else '',
            arch='WRN-28-2'))
        model = WideResNet(depth=28, num_classes=10, widen_factor=2, dropRate=0.2)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    #LOG.info(parameters_string(model))

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
        label_list = torch.zeros(len(dataset.imgs)).long() + 100
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
#         assert len(pred_list) == len(data_idxs)
#         assert len(label_list) == len(data_idxs)
        return pseudo_list, label_list
    
    def get_vibration(pseudo_list, weight, epoch):
        array_size = pseudo_list[:, epoch-50:epoch].shape[1]
        yf_fft = fftpack.fft(pseudo_list[:, epoch-50:epoch], axis=1)
        yf_fft_quadsum = np.sum(np.power(abs(yf_fft), 2), axis=1).reshape(-1, 1)
        yf = np.power(abs(yf_fft), 2) / np.tile(yf_fft_quadsum, array_size)
        yf_half = yf[:, range(int(array_size / 2))] * 2
        vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
        return vibration_measure
    
    def flip_vibration(cLoss_list, weight, epoch):
        array_size = cLoss_list[:, epoch-50:epoch].shape[1]
        yf_fft = fftpack.fft(cLoss_list[:, epoch-50:epoch], axis=1)
        yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
        vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
        return vibration_measure 

    def read_cifar10():
        channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])

        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
        
        eval_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

        traindir = '/cache/cifar10_train'
        dataset_train = torchvision.datasets.ImageFolder(traindir, train_transformation)
        dataset_eval = torchvision.datasets.ImageFolder(traindir, eval_transformation)
        return dataset_train, dataset_eval

    
    dataset_train, dataset_eval = read_cifar10()  
#     data.modify_dataset(dataset_train)
    data.modify_dataset(dataset_eval)
    pseudo_list = torch.zeros((len(unlabeled_idxs), 10, args.epochs))
    label_epoch = torch.zeros((len(unlabeled_idxs), args.epochs)).long()
    for epoch in range(args.start_epoch, args.epochs):
        print('EPOCH:', epoch)
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch, training_log)
        
        #if epoch >= 0:
        #    pseudo_list[:,:,epoch], label_epoch[:, epoch] = get_pc(unlabeled_idxs, dataset_eval, ema_model) 
        #if epoch == args.epochs - 1:
        #    np.save('/dev/shm/100initial_cifar10/stage0/label_epoch_iter0.npy', label_epoch)         
        #    np.save('/dev/shm/100initial_cifar10/stage0/pseudo_list_iter0.npy', pseudo_list)
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
    traindir = '/cache/cifar10_train'
    testdir = '/cache/cifar10_test'

    print(args.labeled_batch_size, args.batch_size)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    labeled_idxs_initial = read_dataset_initial(dataset, 10)
#     rand_select = np.random.choice(sorted(set(range(len(dataset.imgs))) - set(labeled_idxs_initial)), 1000, replace=False)
    labeled_idxs_al_iter1 = np.load('/cache/100initial_cifar10/al_vib2_iter1_stf100/idx_selected_iter1.npy')
    #labeled_idxs_al_iter2 = np.load('/dev/shm/100initial_cifar10/al_vib2_iter3/idx_selected_iter2.npy')
    #labeled_idxs_al_iter3 = np.load('/dev/shm/100initial_cifar10/al_vib2_iter3/idx_selected_iter3.npy')
    #labeled_idxs_al_iter4 = np.load('/dev/shm/100initial_cifar10/al_vib2_iter4/idx_selected_iter4.npy')
    
    #labeled_idxs = labeled_idxs_initial
    #labeled_idxs = np.concatenate((labeled_idxs_initial, rand_select))  
    labeled_idxs = np.concatenate((labeled_idxs_initial, labeled_idxs_al_iter1))#, labeled_idxs_al_iter2, labeled_idxs_al_iter3, labeled_idxs_al_iter4))
    labeled_idxs, unlabeled_idxs = data.relabel_dataset_initial(dataset, labeled_idxs)
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

    return train_loader, test_loader, labeled_idxs, unlabeled_idxs


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
    #for i, ((input, ema_input), (target, img_id)) in enumerate(train_loader): 
    for i, ((input, ema_input), target) in enumerate(train_loader):        
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        
        input_var = input.cuda()
        #weight = weights[img_id].cuda()
        #labels_iter0 = labels_training[img_id]
        with torch.no_grad():
            ema_input_var = ema_input.cuda()
        #target_var = torch.autograd.Variable(target.cuda(async=True))
        target_var = target.cuda()

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        ema_logit, _,_ = ema_model(ema_input_var)
        class_logit, cons_logit,_ = model(input_var)

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)       
#         ema_logit_new = ema_logit.data.cpu()
#         false_label_idx1, false_label_idx2 = np.where(false_target == 0)
#         ema_logit_new[false_label_idx1, false_label_idx2] = -1000000
#         ema_logit_new = ema_logit_new.cuda()

        if args.logit_distance_cost >= 0:
            #class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        
        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            #meters.update('cons_weight', consistency_weight)
            weight = torch.ones(len(ema_logit)).cuda()
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit, weight) / minibatch_size
            #meters.update('cons_loss', consistency_loss.data[0])
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
            target_var = target

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        #meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        with torch.no_grad():
            output1, _,_ = model(input_var)
        
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
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target):
    _, prediction = torch.max(output, dim=1)
    # print(prediction)
    # pdb.set_trace()
    num_correct = np.sum(prediction.numpy() == target.numpy()) #/ len(target)
    return num_correct, len(target)

def unpack_cifar10():
    work_dir = '/cache/'
    test_dir = '/cache/cifar10_test'
    train_dir = '/cache/cifar10_train'

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
    if not os.path.exists('/cache/cifar10_train'):
        src = '.../data/cifar-10-batches-py/'
        dst = '/cache/cifar-10-batches-py'
        mox.file.copy_parallel(src, dst)
        unpack_cifar10()    
    
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    #if os.path.exists('/dev/shm/100initial_cifar10/vib2_iter1_dropout'):
    #    args.resume = '/dev/shm/100initial_cifar10/vib2_iter1_dropout/checkpoint.ckpt'  
    main(RunContext(__file__, 0))
    
