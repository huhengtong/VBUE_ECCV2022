# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms
from RandAugment import RandAugment

from . import data
from .utils import export
#from mean_teacher import custom_transforms as tr


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        #'datadir': 'data-local/images/ilsvrc2012/',
#         'datadir': 's3://bucket-0001/data/ImageNet',
        'num_classes': 1000
    }


@export
def miniimagenet():
    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    #train_transformation = data.TransformTwice(transforms.Compose([
    #    transforms.RandomRotation(10),
    #    transforms.RandomCrop(84, padding=8),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize(**channel_stats)
    #]))
    
    ### randAugment
    train_transformation = transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    N, M = 2, 4
    train_transformation.transforms.insert(0, RandAugment(N, M))    
    train_transformation = data.TransformTwice(train_transformation)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        #'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 100
    }


@export
def cifar100():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    ### randAugment
    train_transformation = transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    N, M = 2, 4
    train_transformation.transforms.insert(0, RandAugment(N, M))    
    train_transformation = data.TransformTwice(train_transformation)
    
    #train_transformation = data.TransformTwice(transforms.Compose([
    #    data.RandomTranslateWithReflect(4),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize(**channel_stats)
    #]))
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        #'datadir': 'data-local/images/cifar/cifar10/by-image',
        #'datadir': '/cache',
        'num_classes': 100
    }

@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    ### randAugment
    #train_transformation = transforms.Compose([
    #    data.RandomTranslateWithReflect(4),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize(**channel_stats)
    #])
    
    #N, M = 1, 5
    #train_transformation.transforms.insert(0, RandAugment(N, M))    
    #train_transformation = data.TransformTwice(train_transformation)
    
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

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        #'datadir': 'data-local/images/cifar/cifar10/by-image',
        #'datadir': '/cache',
        'num_classes': 10
    }
    
    
@export
def VOC2007():
    #train_transformation = data.TransformTwice( transforms.Compose(
    #        [#tr.RandomHorizontalFlip(),
    #        transforms.RandomResizedCrop(224),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #        ]))
            
    ### randAugment
    train_transformation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    N, M = 2, 4
    train_transformation.transforms.insert(0, RandAugment(N, M))    
    train_transformation = data.TransformTwice(train_transformation)
      
    test_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return {
        'train_transformation': train_transformation,
        'eval_transformation': test_transformation,
        'num_classes': 20
    }
  
  
  