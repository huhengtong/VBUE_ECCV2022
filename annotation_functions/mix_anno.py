import numpy as np
import torch
import torchvision.datasets
import torch.nn as nn
import pdb
import torchvision.transforms as transforms
import os
import moxing as mox

import torch.nn.functional as F
from mean_teacher import data, datasets
from mean_teacher.resnet import resnet50
from mean_teacher import architectures
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from kcenter_greedy import kCenterGreedy
from sklearn.metrics import pairwise_distances
import scipy.fftpack as fftpack


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

    traindir = '/dev/shm/cifar100_train'
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

def read_model_resnet(checkpoint_path):
    model = resnet50(num_classes=100)
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

def compare_prediction(data_idxs, dataset, model):
    model.eval()
#     sampler = SubsetRandomSampler(data_idxs)
#     batch_sampler = BatchSampler(sampler, 256, drop_last=False)
    pred_loader = torch.utils.data.DataLoader(dataset,
                                batch_size=512,
                                shuffle=False,
                                #batch_sampler=batch_sampler,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)
    features_list, pred_value_list, idx_list, target_list, pred_label_list = [], [], [], [], []
    H_full_bit_anno_list, H_one_bit_anno_list = [], []
    CE_loss = []
    pred_all_label_list = torch.zeros(len(dataset.imgs)).long()
    for i, (input, (target, idx)) in enumerate(pred_loader):
        input = input.cuda()
        with torch.no_grad():
            output, _,feature = model(input)

        CE_loss.append(F.cross_entropy(output.data.cpu(), target, reduction='none'))
        features_list.append(feature.data.cpu())
        idx_list.append(idx)
        target_list.append(target)
        pred_value, pred = torch.max(torch.softmax(output, dim=1).data.cpu(), dim=1)
        pred_all_label_list[idx] = pred

        pred_value_list.append(pred_value)
        pred_label_list.append(pred)

    CE_loss = torch.cat(CE_loss, dim=0)
    features_list = torch.cat(features_list, dim=0)
    pred_value_list = torch.cat(pred_value_list, dim=0)
    idx_list = torch.cat(idx_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    pred_label_list = torch.cat(pred_label_list, dim=0)   
    return features_list, idx_list, pred_value_list, target_list, pred_label_list, CE_loss

def get_vibration(pseudo_list, weight, best_epoch):
    array_size = pseudo_list[:, 100:160].shape[1]
    yf_fft = fftpack.fft(pseudo_list[:, 100:160], axis=1)
    yf_fft_quadsum = np.sum(np.power(abs(yf_fft), 2), axis=1).reshape(-1, 1)
    yf = np.power(abs(yf_fft), 2) / np.tile(yf_fft_quadsum, array_size)
    yf_half = yf[:, range(int(array_size / 2))] * 2
    vibration_measure = np.sum(yf_half[:, 1:30], axis=1) - weight*yf_half[:, 0] 
    return vibration_measure
    
def flip_vibration(cLoss_list, weight, best_epoch):
    array_size = cLoss_list[:, 100:160].shape[1]
    yf_fft = fftpack.fft(cLoss_list[:, 100:160], axis=1)
    yf_half = abs(yf_fft)[:, range(int(array_size / 2))] * 2
    vibration_measure = np.sum(yf_half[:, 1:30], axis=1) - weight*yf_half[:, 0] 
    return vibration_measure      


def main(name_dataset):
    ### read the data
    if name_dataset == 'cifar': 
        dataset = read_cifar100()    
        labeled_idx_initial = read_dataset_initial(dataset, 30)
        stage1_idx_full_bit = np.load('/dev/shm/index_cifar100/mix_anno_stage1/idx_full.npy')
        stage1_correct_pred_idx = np.load('/dev/shm/index_cifar100/mix_anno_stage1/corPred_idx.npy')
        
#         checkpoint_path = '/dev/shm/mix_anno_new_cifar_stage1/best.ckpt'
        checkpoint_path = '/dev/shm/3000initial_cifar/mix_anno_stage1/best.ckpt'
        model, best_epoch = read_model_shakeshake(checkpoint_path)
        
    elif name_dataset == 'mini': 
        dataset = read_mini()
        labeled_idx_initial = read_dataset_initial(dataset, 30) 
        stage1_idx_full_bit = np.load('/dev/shm/3000initial_mini/mix_anno_stage1/idx_full.npy')
        stage1_correct_pred_idx = np.load('/dev/shm/3000initial_mini/mix_anno_stage1/corPred_idx.npy')

        checkpoint_path = '/dev/shm/3000initial_mini/mix_anno_stage1/best.ckpt'
        model, best_epoch = read_model_resnet(checkpoint_path)
           
#     labeled_idxs = labeled_idx_initial
#     labeled_idxs = np.concatenate((labeled_idx_initial, stage1_select_inds))
    labeled_idxs = np.concatenate((labeled_idx_initial, stage1_idx_full_bit, stage1_correct_pred_idx))
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    print(len(labeled_idxs), len(unlabeled_idxs))
    modify_dataset(dataset)
        
    features_list, idx_list, pred_value_list, target_list, pred_label_list, CE_loss = compare_prediction(range(len(dataset.imgs)), dataset, model)
    print(np.mean(pred_label_list[unlabeled_idxs].numpy() == target_list[unlabeled_idxs].numpy()))
    
    ### FFT SELECT
    pred_label_list = pred_label_list.numpy()[unlabeled_idxs]
    pseudo_list_all = np.load('/dev/shm/3000initial_mini/mix_anno_stage1/pseudo_list_iter1.npy')
    print(pseudo_list_all.shape)
    pseudo_list = np.asarray(pseudo_list_all[range(len(unlabeled_idxs)), pred_label_list, :])
    time_label_list = np.load('/dev/shm/3000initial_mini/mix_anno_stage1/label_epoch_iter1.npy')
    time_label_list = time_label_list.astype(np.int32)
    print(np.mean(time_label_list[:, best_epoch-1] == target_list[unlabeled_idxs].numpy()))
    
    ### confidence vibration
    weight_conf = 0.1
    vibration_conf = get_vibration(pseudo_list, weight_conf, best_epoch) 
    
    ### prediction flip vibration (sharpen the curve)
    pred_flip = np.zeros(time_label_list.shape)
    for i in range(len(time_label_list)):
        pred_flip[i, 0:] = time_label_list[i, 0:] == pred_label_list[i]
    weight_flip = 0.1
    vibration_flip = flip_vibration(pred_flip, weight_flip, best_epoch)    

    ### entropy vibration
    def entropy(prob, base=None):
        result = np.where(prob > 0.0000000001, prob, -10)
        return np.sum(-np.log(result, out=result, where=result>0)*prob, axis=0)
    entropy_list = []
    for i in range(len(unlabeled_idxs)):
        ent = entropy(pseudo_list_all[i])
        entropy_list.append(ent)
    entropy_list = np.stack(entropy_list, axis=0)
    print(entropy_list.shape)
    vibration_ent = flip_vibration(entropy_list, -10, best_epoch)
        
    def normlize(input_list):
            return (input_list-min(input_list))/ (max(input_list)-min(input_list))
        
    vibration_fused = normlize(vibration_conf) + 0.1*normlize(vibration_flip) + 0.1*normlize(vibration_ent)
    inds = np.argsort(vibration_fused)
    ind_full_bit = inds[-1000:]
    ind_one_bit = inds[-32313:-18601]
    
#     kcenter_greedy_aboveTH = kCenterGreedy(torch.cat((features_list[unlabeled_idxs][above_th_inds], features_list[labeled_idxs]), dim=0))
#     labeled_idxs_aboveTH = range(len(above_th_inds), len(above_th_inds)+len(labeled_idxs))
#     selected_inds_aboveTH = kcenter_greedy_aboveTH.select_batch_(labeled_idxs_aboveTH, 18000)
    
#     kcenter_greedy_belowTH = kCenterGreedy(torch.cat((features_list[unlabeled_idxs][below_th_inds], features_list[labeled_idxs]), dim=0))
#     labeled_idxs_aboveTH = range(len(below_th_inds), len(below_th_inds)+len(labeled_idxs))
#     selected_inds_belowTH = kcenter_greedy_belowTH.select_batch_(labeled_idxs_aboveTH, 1355) 
    
    ### find the index for the two kinds of samples, and the corresponding one_bit annotations
    idx_full_bit = idx_list[unlabeled_idxs][ind_full_bit]
    ind_correct_pred = np.where(pred_label_list[ind_one_bit] == target_list[unlabeled_idxs][ind_one_bit].numpy())[0]
    ind_incorrect_pred = np.where(pred_label_list[ind_one_bit] != target_list[unlabeled_idxs][ind_one_bit].numpy())[0]
    correct_pred_idx = idx_list[unlabeled_idxs][ind_one_bit][ind_correct_pred]
    incorrect_pred_idx = idx_list[unlabeled_idxs][ind_one_bit][ind_incorrect_pred]
    incorrect_pred_label = pred_label_list[ind_one_bit][ind_incorrect_pred]
    
    print(len(idx_full_bit), len(correct_pred_idx), len(incorrect_pred_idx), len(incorrect_pred_label))
    pdb.set_trace()
    if name_dataset == 'cifar': 
        os.makedirs('/dev/shm/index_cifar100/mix_anno_stage2', exist_ok=True)
        np.save('/dev/shm/index_cifar100/mix_anno_stage2/idx_full.npy', idx_full_bit)
        np.save('/dev/shm/index_cifar100/mix_anno_stage2/corPred_idx.npy', correct_pred_idx)
        np.save('/dev/shm/index_cifar100/mix_anno_stage2/incorPred_idx.npy', incorrect_pred_idx)
        np.save('/dev/shm/index_cifar100/mix_anno_stage2/incorPred_label.npy', incorrect_pred_label)
    elif name_dataset == 'mini': 
        os.makedirs('/dev/shm/index_mini/mix_anno_stage2', exist_ok=True)
        np.save('/dev/shm/index_mini/mix_anno_stage2/idx_full.npy', idx_full_bit)
        np.save('/dev/shm/index_mini/mix_anno_stage2/corPred_idx.npy', correct_pred_idx)
        np.save('/dev/shm/index_mini/mix_anno_stage2/incorPred_idx.npy', incorrect_pred_idx)
        np.save('/dev/shm/index_mini/mix_anno_stage2/incorPred_label.npy', incorrect_pred_label)
        
    
    
if __name__ == '__main__':
    
    main('mini')   
    
    
    
    
    