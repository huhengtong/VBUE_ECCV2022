# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path
import pdb
import torch
import torchvision

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from mean_teacher import data 


LOG = logging.getLogger('main')
NO_LABEL = -1


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

    
def relabel_dataset_bl_stage1(dataset, labeled_idxs, false_pred_dict):
    for idx in range(len(dataset.imgs)):
        false_target = np.ones(100)
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            dataset.imgs[idx] = path, (label_idx, false_target)
        elif idx in false_pred_dict.keys():
            false_target[false_pred_dict[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target)     
        else:            
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs


def relabel_dataset_bl_stage2(dataset, labeled_idxs, false_pred_dict1, false_pred_dict2):
    for idx in range(len(dataset.imgs)):
        false_target = torch.ones(100)
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
#             if idx in false_pred_dict1.keys():
#                 false_target[false_pred_dict1[idx]] = 0
            dataset.imgs[idx] = path, (label_idx, false_target)
        elif idx in false_pred_dict1.keys():
            false_target[false_pred_dict1[idx]] = 0
            if idx in false_pred_dict2.keys():
                false_target[false_pred_dict2[idx]] = 0       
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
        elif idx in false_pred_dict2.keys() and idx not in false_pred_dict1.keys():
            false_target[false_pred_dict2[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
        else:
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
            
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    return labeled_idxs, unlabeled_idxs


def relabel_dataset_idx(dataset, labeled_idxs):
#     p_labels = torch.from_numpy(p_labels).long()
#     inds_easy = (p_labels > -2).nonzero()
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            dataset.imgs[idx] = path, (label_idx, idx) 
#         elif idx in inds_easy:
#             dataset.imgs[idx] = path, (p_labels[idx], idx)
        else:        
            dataset.imgs[idx] = path, (NO_LABEL, idx)
#     labeled_idxs = np.concatenate((labeled_idxs, inds_easy.squeeze().numpy()))
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs
    

def relabel_dataset_initial(dataset, labeled_idxs):
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]

        if idx in labeled_idxs:
            dataset.imgs[idx] = path, label_idx
        else:
            dataset.imgs[idx] = path, NO_LABEL
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs

def read_imagenet(traindir):
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    #traindir = 'mini-imagenet/train'
    #traindir = args.train_subdir
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset

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

    traindir = '/dev/shm/cifar10_train'
    dataset_train = torchvision.datasets.ImageFolder(traindir, train_transformation)
    dataset_eval = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset_train, dataset_eval

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

def modify_dataset(dataset):
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]
        dataset.imgs[idx] = path, (label_idx, idx)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

# import time
# import faiss
# import scipy
# import torch.nn.functional as F
# from faiss import normalize_L2

def update_plabels(X,labels,labeled_idx,unlabeled_idx, class_num=10, k = 50, max_iter = 20):

    print('Updating pseudo-labels...')
    alpha = 0.99
    labels = np.asarray(labels)
    labeled_idx = np.asarray(labeled_idx)
    unlabeled_idx = np.asarray(unlabeled_idx)

    # kNN search for the graph
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

    normalize_L2(X)
    index.add(X) 
    N = X.shape[0]
    Nidx = index.ntotal

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)

    # Create the graph
    D = D[:,1:] ** 3
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, class_num))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(class_num):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:,i] = f

    # Handle numberical errors
    Z[Z < 0] = 0 

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z),1)#.numpy()
    probs_l1[probs_l1 <0] = 0
    p_values, p_labels = torch.max(probs_l1, dim=1)
    p_labels = p_labels.numpy()
    entropy = scipy.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(class_num)
    weights = weights / np.max(weights)
#     p_labels = np.argmax(probs_l1,1)

    # Compute the accuracy of pseudolabels for statistical purposes
    correct_idx = (p_labels == labels)
    acc = correct_idx.mean()
    inds = np.argsort(weights[unlabeled_idx])[-1000:]
#     inds = np.argsort(p_values)[-1000:]
    acc_easy = np.mean(p_labels[unlabeled_idx[inds]] == labels[unlabeled_idx[inds]])
    print(acc, acc_easy)

    p_labels[labeled_idx] = labels[labeled_idx]
    weights[labeled_idx] = 1.0

#     self.p_weights = weights.tolist()
#     self.p_labels = p_labels

#     # Compute the weight for each class
#     for i in range(len(self.classes)):
#         cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
#         self.class_weights[i] = (float(labels.shape[0]) / len(self.classes)) / cur_idx.size

    return p_labels, inds, weights
