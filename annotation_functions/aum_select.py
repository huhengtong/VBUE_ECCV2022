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
#from mean_teacher.WRNet_mini import WideResNet

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

        CE_loss.append(F.cross_entropy(output.data.cpu(), target, reduction='none'))        
        features_list.append(feature.data.cpu())
        idx_list.append(idx)
        target_list.append(target)
        output_softmax = torch.softmax(output, dim=1).data.cpu()
        _, pred = torch.max(output_softmax, dim=1)
        #pred_value, pred = torch.max(torch.softmax(output, dim=1).data.cpu(), dim=1)
        pred_value_list.append(output_softmax)
        pred_label_list.append(pred)

    features_list = torch.cat(features_list, dim=0)
    pred_value_list = torch.cat(pred_value_list, dim=0)
    idx_list = torch.cat(idx_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    pred_label_list = torch.cat(pred_label_list, dim=0)
    CE_loss = torch.cat(CE_loss, dim=0)
    return features_list, idx_list, pred_value_list, target_list, pred_label_list, CE_loss


def get_vibration(pseudo_list, weight, best_epoch):
    array_size = pseudo_list[:, 100:best_epoch].shape[1]
    yf_fft = fftpack.fft(pseudo_list[:, 100:best_epoch], axis=1)
#     yf_fft_quadsum = np.sum(np.power(abs(yf_fft), 2), axis=1).reshape(-1, 1)
#     yf = np.power(abs(yf_fft), 2) / np.tile(yf_fft_quadsum, array_size)
#     yf_half = yf[:, range(int(array_size / 2))] * 2
    yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
    vibration_measure = np.sum(yf_half[:, 1:20], axis=1) - weight*yf_half[:, 0] 
    return vibration_measure
    
def flip_vibration(cLoss_list, weight, best_epoch):
    array_size = cLoss_list[:, 100:best_epoch].shape[1]
    yf_fft = fftpack.fft(cLoss_list[:, 100:best_epoch], axis=1)
    yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
    vibration_measure = np.sum(yf_half[:, 1:], axis=1) - weight*yf_half[:, 0] 
    return vibration_measure                   
    
def normlize(input_list):
    return (input_list-min(input_list))/ (max(input_list)-min(input_list))
    
def main(name_dataset):
    ### read the data
    if name_dataset == 'cifar': 
        #dataset = read_cifar10()  
        dataset = read_cifar100()    
        labeled_idx_initial = read_dataset_initial(dataset, 50)
        #stage1_idx_full_bit = np.load('/dev/shm/5000initial_dropout/al_aum_iter1/idx_selected_iter1.npy')
        #stage2_idx_full_bit = np.load('/dev/shm/5000initial_dropout/al_aum_iter2/idx_selected_iter2.npy')
        #stage3_idx_full_bit = np.load('/dev/shm/5000initial_dropout/al_aum_iter3/idx_selected_iter3.npy')
#         stage1_correct_pred_idx = np.load('/dev/shm/index_cifar100/mix_annotation/stage1_correct_pred_idx.npy')
        
        checkpoint_path = '/dev/shm/5000initial_dropout/stage0_dropout/best.ckpt'
        #checkpoint_path = '/dev/shm/5000initial_dropout/al_aum_iter3/best.ckpt'
        model, best_epoch = read_model_WRN(checkpoint_path, num_classes=100)  
    elif name_dataset == 'mini': 
        dataset = read_mini()
        labeled_idx_initial = read_dataset_initial(dataset, 100) 
        stage1_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_aum_iter2/idx_selected_iter1.npy')
        stage2_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_aum_iter2/idx_selected_iter2.npy')
        stage3_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_aum_iter3/idx_selected_iter3.npy')
        checkpoint_path = '/dev/shm/10000initial_mini/al_aum_iter3/best.ckpt'
        #model, best_epoch = read_model_WRN(checkpoint_path, num_classes=100)
           
    labeled_idxs = labeled_idx_initial
    #labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage2_idx_full_bit, stage3_idx_full_bit))
#     labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage1_correct_pred_idx))
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    unlabeled_idxs = np.asarray(unlabeled_idxs)
    print(len(labeled_idxs), len(unlabeled_idxs))
    modify_dataset(dataset)
    
    features_list, idx_list, pred_value_list, target_list, pred_label_list, CE_loss = compare_prediction(range(len(dataset.imgs)), dataset, model)
    
    pred_label_list = pred_label_list.numpy()[unlabeled_idxs]
    #best_epoch = 179
    preds_list = np.load('/dev/shm/5000initial_dropout/stage0_dropout/pseudo_list_iter0.npy')
    print(preds_list.shape)
    #pred_value_list, pred_label_list = torch.max(torch.from_numpy(preds_list[:, :, best_epoch]), dim=1)
    #pred_label_list = pred_label_list.numpy()
    
    pseudo_list = np.asarray(preds_list[range(len(unlabeled_idxs)), pred_label_list, :])
    sec_max_list = []
    for i in range(len(unlabeled_idxs)):
        preds_dele = np.delete(preds_list[i], pred_label_list[i], axis=0)      
        sec_max_list.append(np.max(preds_dele, axis=0))
    sec_max_list = np.stack(sec_max_list)
    print(sec_max_list.shape)
    
    aum = np.mean((pseudo_list[:, 100:best_epoch] - sec_max_list[:, 100:best_epoch]), axis=1)
    #aum = normlize(aum)
    print('max:', max(aum), 'min:', min(aum))
    
    
    inds_selected = np.argsort(aum)[-5000:]
    ece_scores = ece_score(pred_value_list[unlabeled_idxs][inds_selected], target_list[unlabeled_idxs][inds_selected], 10)
    print(ece_scores)
    pdb.set_trace()
    
    
    
    
    ind_full_bit = np.argsort(aum)[0:2500]
    print(aum[ind_full_bit][0:10])
    idx_selected = unlabeled_idxs[ind_full_bit]  
    #idx_or = idx_list[unlabeled_idxs][np.argsort(vibration_conf)[-500:]]
    #print(len(set(idx_or.numpy()) & set(idx_selected.numpy())))
        
    print(len(idx_selected))
    pdb.set_trace()
    os.makedirs('/dev/shm/10000initial_mini/al_aum_iter4', exist_ok=True)
    np.save('/dev/shm/10000initial_mini/al_aum_iter4/idx_selected_iter4.npy', idx_selected) 
    
    
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
    if not os.path.exists('/dev/shm/cifar10_train'):
        src = '/data/VBUE/data/cifar-10-batches-py/'
        dst = '/dev/shm/cifar-10-batches-py'
        mox.file.copy_parallel(src, dst)
        unpack_cifar10()    
    #if not os.path.exists('/dev/shm/5000initial_dropout/al_aum_iter1'):
    
    main('cifar')   
    
    
    
    
    