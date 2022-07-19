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
    #dataset = torchvision.datasets.cifar100
    for idx in range(len(dataset.imgs)):
        path, label = dataset.imgs[idx]
        if label in data_dict.keys():
            data_dict[label].append(idx)
        else:
            data_dict[label] = [idx]

    data_selected = np.array(list(data_dict.values()))
    #print(data_selected.shape)
    labeled_data = []
    for elem in data_selected:
        #ind = np.random.permutation(500)[0:num]
        labeled_data.append(np.array(elem[0:num]))
    #print(np.concatenate(labeled_data).shape)
    #pdb.set_trace()
    return np.concatenate(labeled_data)

def create_data_loaders_initial(train_transformation,
                        eval_transformation,
                        args):
#     traindir = '/dev/shm/mini_imagenet/train'
#     testdir = '/dev/shm/mini_imagenet/test'
    traindir = '/cache/cifar100_train'
    testdir = '/cache/cifar100_test'
#     traindir = args.train_subdir
#     testdir = args.eval_subdir

    print(args.labeled_batch_size, args.batch_size)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    labeled_idxs = read_dataset_initial(dataset, 30)
    labeled_idxs, unlabeled_idxs = relabel_dataset_initial(dataset, labeled_idxs)
    print(len(labeled_idxs), len(unlabeled_idxs))
#     pdb.set_trace()

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


def relabel_dataset_initial(dataset, labeled_idxs):
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            dataset.imgs[idx] = path, label_idx
        else:
            dataset.imgs[idx] = path, NO_LABEL
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    return labeled_idxs, unlabeled_idxs
