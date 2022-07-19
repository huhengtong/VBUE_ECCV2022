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
from mean_teacher.WRNet_mini import WideResNet


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

    traindir = '/dev/shm/cifar100_train'
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

    traindir = '/cache/mini_imagenet/train'
    #traindir = args.train_subdir
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset

def read_model_resnet(checkpoint_path, num_classes):
#     model = resnet50(num_classes=100)
#     model = ResNet18(num_classes)
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
    best_epoch = checkpoint['epoch']
    print('epoch:', best_epoch, 'best_acc', best_prec1)
    model.load_state_dict(checkpoint['ema_state_dict'])
    return model, best_epoch

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
            output, _,_ = model(input)

#         log_softmax_output = F.log_softmax(output.data.cpu(), dim=1)
#         softmax_output = torch.softmax(output, dim=1).data.cpu()
#         H_full_bit_anno = torch.sum(-log_softmax_output*softmax_output, dim=1)
        CE_loss.append(F.cross_entropy(output.data.cpu(), target, reduction='none'))        
#         features_list.append(feature.data.cpu())
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
#     features_list = torch.cat(features_list, dim=0)
    pred_value_list = torch.cat(pred_value_list, dim=0)
    idx_list = torch.cat(idx_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    pred_label_list = torch.cat(pred_label_list, dim=0)
    CE_loss = torch.cat(CE_loss, dim=0)
    return idx_list, pred_value_list, target_list, pred_label_list, CE_loss

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
    
# def vibration_std_mean(input_list, weight, best_epoch):
#     std = np.std(input_list[:, best_epoch-30:best_epoch+30], axis=1)
#     mean = np.mean(input_list[:, best_epoch-30:best_epoch+30], axis=1)
#     list_moments = np.stack((mean, std), axis=1)
# #     list_quadsum = np.sum(np.power(list_moments, 2), axis=1).reshape(-1, 1)
# #     list_moments = np.power(list_moments, 2) / np.tile(list_quadsum, 2)
# #     if norm:
#     list_max = np.max(list_moments, axis=1).reshape(-1, 1)
#     list_moments = list_moments / np.tile(list_max, 2)
#     vibration_measure = list_moments[:, 1] - weight*list_moments[:, 0]
#     return vibration_measure
    
def normlize(input_list):
    return (input_list-min(input_list))/ (max(input_list)-min(input_list))
    
def main(name_dataset):
    ### read the data
    if name_dataset == 'mini': 
        dataset = read_mini()
        labeled_idx_initial = read_dataset_initial(dataset, 100) 
        stage1_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_vib1_iter1/idx_selected_iter1.npy')
        stage2_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_vib1_iter2/idx_selected_iter2.npy')
        stage3_idx_full_bit = np.load('/dev/shm/10000initial_mini/al_vib1_iter3/idx_selected_iter3.npy')
        
        checkpoint_path = '/dev/shm/10000initial_mini/al_vib1_iter3/best.ckpt'
        model, best_epoch = read_model_WRN(checkpoint_path, num_classes=100)
        #if best_epoch > 160:
        #    best_epoch = 150
            
    #labeled_idxs = labeled_idx_initial
    labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage2_idx_full_bit, stage3_idx_full_bit))
#     labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage1_correct_pred_idx))
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    unlabeled_idxs = np.asarray(unlabeled_idxs)
    print(len(labeled_idxs), len(unlabeled_idxs))
    modify_dataset(dataset)
    
    idx_list, pred_value_list, target_list, pred_label_list, _ = compare_prediction(range(len(dataset.imgs)), dataset, model)
    
    #pred_label_list = pred_label_list.numpy()[unlabeled_idxs]
    pseudo_list_all = np.load('/dev/shm/10000initial_mini/al_vib1_iter3/pseudo_list_iter3.npy')
    print(pseudo_list_all.shape)
    
    _, pred_label_list = torch.max(torch.from_numpy(pseudo_list_all[:, :, best_epoch-1]), dim=1)
    pred_label_list = pred_label_list.numpy()
    
    pseudo_list = np.asarray(pseudo_list_all[range(len(unlabeled_idxs)), pred_label_list, :])
    
    time_label_list = np.load('/dev/shm/10000initial_mini/al_vib1_iter3/label_epoch_iter3.npy')
    time_label_list = time_label_list.astype(np.int32)
    #print(np.mean(time_label_list[:, best_epoch-1] == target_list[unlabeled_idxs].numpy()))
 
    weight_conf = 0.1
    vibration_conf = get_vibration(pseudo_list, weight_conf, best_epoch) 
    #ind_full_bit = np.argsort(vibration_conf)[-2500:]
    #idx_selected = idx_list[unlabeled_idxs][ind_full_bit]
    
#     ### prediction flip vibration (sharpen the curve)
    pred_flip = np.zeros(time_label_list.shape)
    for i in range(len(time_label_list)):
        pred_flip[i, 0:] = time_label_list[i, 0:] == pred_label_list[i]
    weight_flip = 0.1
    vibration_flip = flip_vibration(pred_flip, weight_flip, best_epoch)    
               
    vibration_fused = 0.4*normlize(vibration_conf) + 0.6*normlize(vibration_flip) #+ 0.1*normlize(vibration_ent)
    ind_full_bit = np.argsort(vibration_fused)[-2500:]
    idx_selected = unlabeled_idxs[ind_full_bit] 
        
    print(len(idx_selected))
    pdb.set_trace()
    os.makedirs('/dev/shm/10000initial_mini/al_vib1_iter4', exist_ok=True)
    np.save('/dev/shm/10000initial_mini/al_vib1_iter4/idx_selected_iter4.npy', idx_selected) 
    
    
if __name__ == '__main__':
    if not os.path.exists('/cache/mini_imagenet'):
        src = '.../data/mini_imagenet/'
        dst = '/cache/mini_imagenet'
        mox.file.copy_parallel(src, dst)

    main('mini')   
    
    
    
    
    