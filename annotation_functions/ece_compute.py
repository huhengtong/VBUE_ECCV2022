import numpy as np
import torch
import torchvision.datasets
import torch.nn as nn
import pdb
import torchvision.transforms as transforms
import os

import torch.nn.functional as F
from mean_teacher import data, datasets
from mean_teacher.resnet import resnet50
from mean_teacher import architectures
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from kcenter_greedy import kCenterGreedy
import moxing as mox
from sklearn.metrics import pairwise_distances
import scipy.fftpack as fftpack
from mean_teacher.resnet_cifar10 import ResNet18
from mean_teacher.WRNet import WideResNet

import pickle
import sys
from torchvision.datasets import CIFAR100, CIFAR10
import matplotlib.image
import scipy.fftpack as fftpack
from ECE import ece_score


def read_dataset_initial(dataset, num):
    data_dict = {}
    #dataset = torchvision.datasets.cifar100
    for idx in range(len(dataset.imgs)):
        path, label = dataset.imgs[idx]
        if label in data_dict.keys():
            data_dict[label].append(idx)
        else:
            data_dict[label] = [idx]

    data_selected = np.array(list(data_dict.values()))
    labeled_data = []
    for elem in data_selected:
        #ind = np.random.permutation(500)[0:num]
        labeled_data.append(np.array(elem[:num]))
    return np.concatenate(labeled_data)

def read_cifar100():
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

    traindir = '/cache/cifar100_train'
    dataset_train = torchvision.datasets.ImageFolder(traindir, train_transformation)
    dataset_eval = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset_train, dataset_eval
def read_cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    traindir = '/dev/shm/cifar10_train'
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset


def read_mini():
    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    traindir = '/dev/shm/mini_imagenet/train'
    #traindir = args.train_subdir
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset

def read_model_resnet(checkpoint_path, num_classes):
#     model = resnet50(num_classes=100)
    model = ResNet18(num_classes=num_classes)
    model = nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    best_prec1 = checkpoint['best_prec1']
    best_epoch = checkpoint['epoch']
    print('epoch:', best_epoch, 'best_acc', best_prec1)
    model.load_state_dict(checkpoint['ema_state_dict'])
    return model, best_epoch
  
def read_model_WRN(checkpoint_path, num_classes):
#     model = resnet50(num_classes=100)
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.0)
    model = nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    best_prec1 = checkpoint['best_prec1']
    best_epoch = checkpoint['epoch']
    print('epoch:', best_epoch, 'best_acc', best_prec1)
    model.load_state_dict(checkpoint['ema_state_dict'])
    return model, best_epoch

def read_model_shakeshake(checkpoint_path):
    model_factory = architectures.__dict__['cifar_shakeshake26']
    model_params = dict(num_classes=100)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()
    
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    best_prec1 = checkpoint['best_prec1']
    print('best_acc', best_prec1)
    model.load_state_dict(checkpoint['ema_state_dict'])
    return model

def modify_dataset(dataset):
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]
        dataset.imgs[idx] = path, (label_idx, idx)

def compute_cLoss(data_idxs, dataset, ema_model):
        ema_model.eval()
        sampler = SubsetRandomSampler(data_idxs)
        batch_sampler = BatchSampler(sampler, 512, drop_last=False)
        pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=8,
                                pin_memory=True)
        closs_list = torch.zeros(len(dataset.imgs))
        for i, ((input, ema_input), (target, idx)) in enumerate(pred_loader):
            input, ema_input = input.cuda(), ema_input.cuda()
            with torch.no_grad():
                output,_,_ = ema_model(input)
                output_ema,_,_ = ema_model(ema_input)
                
            output_sm = torch.softmax(output.data.cpu(), dim=1)
            output_ema_sm = torch.softmax(output_ema.data.cpu(), dim=1)
            c_loss = F.pairwise_distance(output_sm, output_ema_sm, p=2)
            closs_list[idx] = c_loss
        closs_list = closs_list[data_idxs]
        return closs_list   
        
def compare_prediction(data_idxs, dataset, model):
    model.eval()
    #sampler = SubsetRandomSampler(data_idxs)
    #batch_sampler = BatchSampler(sampler, 256, drop_last=False)
    pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_size=200,
                                shuffle=False,
                                #batch_sampler=batch_sampler,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)
    pred_value_list, idx_list, target_list, pred_label_list = [], [], [], []
    
    for i, (input, (target, idx)) in enumerate(pred_loader):
        input = input.cuda()
        with torch.no_grad():
            output, _,feature = model(input)

        idx_list.append(idx)
        target_list.append(target)
        output_softmax = torch.softmax(output, dim=1).data.cpu()
        _, pred = torch.max(output_softmax, dim=1)
        
        pred_value_list.append(output_softmax)
        pred_label_list.append(pred)

    pred_value_list = torch.cat(pred_value_list, dim=0)
    idx_list = torch.cat(idx_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    pred_label_list = torch.cat(pred_label_list, dim=0)
    return idx_list, pred_value_list, target_list, pred_label_list

def get_vibration(pseudo_list, weight, best_epoch):
    array_size = pseudo_list[:, 100:best_epoch].shape[1]
    yf_fft = fftpack.fft(pseudo_list[:, 100:best_epoch], axis=1)
#     yf_fft_quadsum = np.sum(np.power(abs(yf_fft), 2), axis=1).reshape(-1, 1)
#     yf = np.power(abs(yf_fft), 2) / np.tile(yf_fft_quadsum, array_size)
#     yf_half = yf[:, range(int(array_size / 2))] * 2
    yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
    vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
    return vibration_measure
    
def flip_vibration(cLoss_list, weight, best_epoch):
    array_size = cLoss_list[:, 100:best_epoch].shape[1]
    yf_fft = fftpack.fft(cLoss_list[:, 100:best_epoch], axis=1)
    yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
    vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
    return vibration_measure    
    
def normlize(input_list):
    return (input_list-min(input_list))/ (max(input_list)-min(input_list))
    
def entropy(prob, base=None):
    result = np.where(prob > 0.0000000001, prob, -10)
    return np.sum(-np.log(result, out=result, where=result>0)*prob, axis=0)
    
def main(name_dataset):
    ### read the data
    if name_dataset == 'cifar': 
        #dataset_eval = read_cifar10()  
        _, dataset_eval = read_cifar100()    
        labeled_idx_initial = read_dataset_initial(dataset_eval, 50)
        #stage1_idx_full_bit = np.load('/dev/shm/100initial_cifar10/al_vib2_iter1/idx_selected_iter1.npy')
#         stage2_idx_full_bit = np.load('/dev/shm/3000initial_cifar10/fused_fft_AL_iter2/idx_selected_iter2.npy')
#         stage3_idx_full_bit = np.load('/dev/shm/3000initial_cifar10/fused_fft_AL_iter3/idx_selected_iter3.npy')
#         stage1_correct_pred_idx = np.load('/dev/shm/index_cifar100/mix_annotation/stage1_correct_pred_idx.npy')
        
#         checkpoint_path = '/dev/shm/3000initial_cifar10/fft_AL_iter3/best.ckpt'
        checkpoint_path = '/dev/shm/5000initial_dropout/stage0_dropout/best.ckpt'
        model, best_epoch = read_model_WRN(checkpoint_path, num_classes=100)   
        #best_epoch = 150
           
    labeled_idxs = labeled_idx_initial
#     labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage2_idx_full_bit, stage3_idx_full_bit))
#     labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage1_correct_pred_idx))
    unlabeled_idxs = sorted(set(range(len(dataset_eval.imgs))) - set(labeled_idxs))
    print(len(labeled_idxs), len(unlabeled_idxs))
    modify_dataset(dataset_eval)
    
    idx_list, pred_value_list, target_list, pred_label_list = compare_prediction(range(len(dataset_eval.imgs)), dataset_eval, model)
    #modify_dataset(dataset_train)
    #consistency_list = compute_cLoss(unlabeled_idxs, dataset_train, model)
    
    pred_label_list = pred_label_list.numpy()[unlabeled_idxs]
    pseudo_list = np.load('/dev/shm/5000initial_dropout/stage0_dropout/pseudo_list_iter0.npy')
    print(pseudo_list.shape)
    pseudo_list = np.asarray(pseudo_list[range(len(unlabeled_idxs)), pred_label_list, :])
    
    weight_conf = 0.1
    vibration_conf = get_vibration(pseudo_list, weight_conf, best_epoch) 
    #inds_selected = np.argsort(vibration_conf)[0:5000]
    
    time_label_list = np.load('/dev/shm/5000initial_dropout/stage0_dropout/label_epoch_iter0.npy')
    time_label_list = time_label_list.astype(np.int32)
    
    ### prediction flip vibration (sharpen the curve)
    pred_flip = np.zeros(time_label_list.shape)
    for i in range(len(time_label_list)):
        pred_flip[i, 0:] = time_label_list[i, 0:] == pred_label_list[i]
    
    weight_flip = 0.1
    vibration_flip = flip_vibration(pred_flip, weight_flip, best_epoch)
    
    vibration_fused = 0.8*normlize(vibration_conf) + 0.2*normlize(vibration_flip)
    print('max:', max(vibration_fused), 'min:', min(vibration_fused))
    
    #pred_prob = pred_value_list[unlabeled_idxs][range(len(unlabeled_idxs)), pred_label_list]
    #entropy_list = []
    #for i in range(len(unlabeled_idxs)):
    #    ent = entropy(pred_value_list[unlabeled_idxs][i].numpy())
    #    entropy_list.append(ent)
    #print(len(entropy_list))
    
    #print(consistency_list.size())
    inds_selected = np.argsort(vibration_fused)[0:10000]
    #inds_selected = np.where(vibration_fused <= 1.0)
    #print(pred_prob[inds_selected][0:10])
    
    print(np.mean(pred_label_list[inds_selected] == target_list[unlabeled_idxs][inds_selected].numpy()))
    
    ece_scores = ece_score(pred_value_list[unlabeled_idxs][inds_selected], target_list[unlabeled_idxs][inds_selected], 10)
    print(ece_scores)
    pdb.set_trace()
    
    
def unpack_cifar100():
    work_dir = '/dev/shm/'
    test_dir = '/dev/shm/cifar100_test'
    train_dir = '/dev/shm/cifar100_train'

    def load_file(file_name):
        with open(os.path.join(work_dir, 'cifar-100-python', file_name), 'rb') as meta_f:
            return pickle.load(meta_f, encoding="latin1")

    def unpack_data_file(source_file_name, target_dir, start_idx):
        print("Unpacking {} to {}".format(source_file_name, target_dir))
        data = load_file(source_file_name)
        for idx, (image_data, label_idx) in enumerate(zip(data['data'], data['fine_labels'])):#, total=len(data['data']):
            subdir = os.path.join(target_dir, label_names[label_idx])
            name = "{}_{}.png".format(start_idx + idx, label_names[label_idx])
            os.makedirs(subdir, exist_ok=True)
            image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
            matplotlib.image.imsave(os.path.join(subdir, name), image)
        return len(data['data'])

    label_names = load_file('meta')['fine_label_names']
    print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))

    start_idx = 0
    for source_file_path in ['test']:
        start_idx += unpack_data_file(source_file_path, test_dir, start_idx)
    start_idx = 0
    for source_file_path in ['train']:
        start_idx += unpack_data_file(source_file_path, train_dir, start_idx)


    
if __name__ == '__main__':
    if not os.path.exists('/cache/cifar100_train'):
        src = '.../data/cifar-100-python/'
        dst = '/dev/shm/cifar-100-python'
        mox.file.copy_parallel(src, dst)
        unpack_cifar100()    

    main('cifar')   
    
    
    
    
    