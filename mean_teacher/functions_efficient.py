import numpy as np
import torchvision
import torch
import pdb

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from mean_teacher import data
from mean_teacher.utils import *


NO_LABEL = -1
def read_dataset_initial(dataset, num):
    data_dict = {}
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
        labeled_data.append(np.array(elem[0:num]))
    return np.concatenate(labeled_data)


def create_data_loaders_efficient(train_transformation,
                        eval_transformation,
                        args):
#     traindir = '/cache/mini_imagenet/train'
#     testdir = '/cache/mini_imagenet/test'
    traindir = '/dev/shm/cifar100_train'
    testdir = '/dev/shm/cifar100_test'

    print(args.labeled_batch_size, args.batch_size)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    labeled_idxs_initial = read_dataset_initial(dataset, 30)
    one_bit_labeled_idxs_stage1 = np.load('/dev/shm/index_cifar100/efficient_anno/pred_true_stage1.npy')
    full_bit_labeled_idxs_stage1 = np.load('/dev/shm/index_cifar100/efficient_anno/hard_inds_stage1.npy')
    one_bit_labeled_idxs_stage2 = np.load('/dev/shm/index_cifar100/efficient_anno/pred_true_stage2.npy')
    full_bit_labeled_idxs_stage2 = np.load('/dev/shm/index_cifar100/efficient_anno/hard_inds_stage2.npy')
    
#     easy_idxs_stage1 = np.load('/dev/shm/index_cifar100/efficient_anno/easy_inds_stage1.npy')
#     eaasy_preds_stage1 = np.load('/dev/shm/index_cifar100/efficient_anno/easy_preds_stage1.npy')
    pred_false_stage1 = np.load('/dev/shm/index_cifar100/efficient_anno/pred_false_stage1.npy')
    false_pred_stage1 = np.load('/dev/shm/index_cifar100/efficient_anno/false_pred_stage1.npy') 
    pred_false_stage2 = np.load('/dev/shm/index_cifar100/efficient_anno/pred_false_stage2.npy')
    false_pred_stage2 = np.load('/dev/shm/index_cifar100/efficient_anno/false_pred_stage2.npy') 

    labeled_idxs = np.concatenate((labeled_idxs_initial, one_bit_labeled_idxs_stage1, full_bit_labeled_idxs_stage1, one_bit_labeled_idxs_stage2, full_bit_labeled_idxs_stage2))
#     easy_pred_dict1 = dict(zip(easy_idxs_stage1, eaasy_preds_stage1))  
    false_pred_dict1 = dict(zip(pred_false_stage1, false_pred_stage1)) 
    false_pred_dict2 = dict(zip(pred_false_stage2, false_pred_stage2)) 
    labeled_idxs, unlabeled_idxs = relabel_dataset_efficient_stage2(dataset, labeled_idxs, false_pred_dict1, false_pred_dict2)
    print(len(labeled_idxs), len(unlabeled_idxs))
    pdb.set_trace()

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
        num_workers=args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, test_loader


def relabel_dataset_efficient_stage1(dataset, labeled_idxs, false_pred_dict):
    for idx in range(len(dataset.imgs)):
        false_target = np.ones(100)
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            dataset.imgs[idx] = path, (label_idx, false_target)
#         elif idx in easy_pred_dict.keys(): 
#             label_idx = easy_pred_dict[idx]
#             dataset.imgs[idx] = path, (label_idx, false_target)
        elif idx in false_pred_dict.keys():
            false_target[false_pred_dict[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target) 
        else:
            dataset.imgs[idx] = path, (NO_LABEL, false_target) 
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs

def relabel_dataset_efficient_stage2(dataset, labeled_idxs, false_pred_dict1, false_pred_dict2):
    for idx in range(len(dataset.imgs)):
        false_target = np.ones(100)
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            dataset.imgs[idx] = path, (label_idx, false_target)
#         elif idx in easy_pred_dict.keys(): 
#             label_idx = easy_pred_dict[idx]
#             dataset.imgs[idx] = path, (label_idx, false_target)
        elif idx in false_pred_dict1.keys():
            false_target[false_pred_dict1[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target) 
        elif idx in false_pred_dict2.keys():
            false_target[false_pred_dict2[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target) 
        else:
            dataset.imgs[idx] = path, (NO_LABEL, false_target) 
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs