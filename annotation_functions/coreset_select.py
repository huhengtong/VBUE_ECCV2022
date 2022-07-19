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
from mean_teacher.resnet18_mini import resnet18
#from mean_teacher.WRNet import WideResNet
from mean_teacher.WRNet_mini import WideResNet

import pickle
import sys
from torchvision.datasets import CIFAR100
import matplotlib.image


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

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    traindir = '/cache/cifar100_train'
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset

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

    traindir = '/cache/mini_imagenet/train'
    #traindir = args.train_subdir
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset

def read_model_resnet(checkpoint_path, num_classes):
#     model = resnet50(num_classes=100)
#     model = ResNet18(num_classes=num_classes)
    model = resnet18(num_classes=num_classes)
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
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.2)
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

def compare_prediction(data_idxs, dataset, model):
    model.eval()
#     sampler = SubsetRandomSampler(data_idxs)
#     batch_sampler = BatchSampler(sampler, 256, drop_last=False)
    pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_size=200,
                                shuffle=False,
                                #batch_sampler=batch_sampler,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)
    features_list, pred_value_list, idx_list, target_list, pred_label_list = [], [], [], [], []
    H_full_bit_anno_list, H_one_bit_anno_list = [], []
    CE_loss = []
    for i, (input, (target, idx)) in enumerate(pred_loader):
        input = input.cuda()
        with torch.no_grad():
            output, _,feature = model(input)

#         log_softmax_output = F.log_softmax(output.data.cpu(), dim=1)
#         softmax_output = torch.softmax(output, dim=1).data.cpu()
#         H_full_bit_anno = torch.sum(-log_softmax_output*softmax_output, dim=1)
        CE_loss.append(F.cross_entropy(output.data.cpu(), target, reduction='none'))        
        features_list.append(feature.data.cpu())
        idx_list.append(idx)
        target_list.append(target)
        pred_value, pred = torch.max(torch.softmax(output, dim=1).data.cpu(), dim=1)
#         print(torch.softmax(output, dim=1)[0], target[0], CE_loss[0][0])
#         pdb.set_trace()
#         H_one_bit_anno = F.binary_cross_entropy(pred_value, pred_value, reduction='none')
#         _, top3_preds = torch.topk(output.data.cpu(), k=3, dim=1)

        pred_value_list.append(pred_value)
        pred_label_list.append(pred)
#         top3_pred_label_list.append(top3_preds)
#         H_full_bit_anno_list.append(H_full_bit_anno)
#         H_one_bit_anno_list.append(H_one_bit_anno)

#     H_full_bit_anno_list = torch.cat(H_full_bit_anno_list, dim=0)
#     H_one_bit_anno_list = torch.cat(H_one_bit_anno_list, dim=0)
    features_list = torch.cat(features_list, dim=0)
    pred_value_list = torch.cat(pred_value_list, dim=0)
    idx_list = torch.cat(idx_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    pred_label_list = torch.cat(pred_label_list, dim=0)
    CE_loss = torch.cat(CE_loss, dim=0)
    return features_list, idx_list, pred_value_list, target_list, pred_label_list, CE_loss


def main(name_dataset):
    ### read the data
    if name_dataset == 'cifar': 
        #dataset = read_cifar10()  
        dataset = read_cifar100()    
        labeled_idx_initial = read_dataset_initial(dataset, 50)
        stage1_idx_full_bit = np.load('/dev/shm/5000initial_cifar100/al_coreset_iter1/idx_selected_iter1.npy')
        stage2_idx_full_bit = np.load('/dev/shm/5000initial_cifar100/al_coreset_iter2/idx_selected_iter2.npy')
        stage3_idx_full_bit = np.load('/dev/shm/5000initial_cifar100/al_coreset_iter3/idx_selected_iter3.npy')
        #stage4_idx_full_bit = np.load('/dev/shm/5000initial_cifar100/al_coreset_iter4/idx_selected_iter4.npy')
        
        checkpoint_path = '/dev/shm/5000initial_cifar100/al_coreset_iter3/best.ckpt'
        model, best_epoch = read_model_WRN(checkpoint_path, num_classes=100)
#         model = read_model_shakeshake(checkpoint_path)
        
    elif name_dataset == 'mini': 
        dataset = read_mini()
        labeled_idx_initial = read_dataset_initial(dataset, 100) 
        stage1_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_coreset_iter1/idx_selected_iter1.npy')
        stage2_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_coreset_iter2/idx_selected_iter2.npy')
        stage3_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_coreset_iter3/idx_selected_iter3.npy')
        checkpoint_path = '/dev/shm/10000initial_mini/al_coreset_iter3/best.ckpt'
        model, best_epoch = read_model_WRN(checkpoint_path, num_classes=100)
           
    #labeled_idxs = labeled_idx_initial
    labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage2_idx_full_bit, stage3_idx_full_bit))
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    print(len(labeled_idxs), len(unlabeled_idxs))
    modify_dataset(dataset)
    
    features_list, idx_list, pred_value_list, target_list, pred_label_list, CE_loss = compare_prediction(range(len(dataset.imgs)), dataset, model)
    print(np.mean(pred_label_list[unlabeled_idxs].numpy() == target_list[unlabeled_idxs].numpy()))
    
    ### k-center greedy 
    kcenter_greedy_my = kCenterGreedy(features_list)
    selected_inds = kcenter_greedy_my.select_batch_(labeled_idxs, 2500)
    idx_selected = idx_list[selected_inds]
    
    print(len(idx_selected))
    pdb.set_trace()
    os.makedirs('/dev/shm/10000initial_mini/al_coreset_iter4', exist_ok=True)
    np.save('/dev/shm/10000initial_mini/al_coreset_iter4/idx_selected_iter4.npy', idx_selected)    
    
    
def unpack_cifar100():
    work_dir = '/cache/'
    test_dir = '/cache/cifar100_test'
    train_dir = '/cache/cifar100_train'

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
        dst = '/cache/cifar-100-python'
        mox.file.copy_parallel(src, dst)
        unpack_cifar100()    

    main('mini')   
    
   