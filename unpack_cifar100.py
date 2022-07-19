# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import os
import pickle
import sys

#from tqdm import tqdm
from torchvision.datasets import CIFAR100
import matplotlib.image
import numpy as np


# work_dir = os.path.abspath(sys.argv[1])
# test_dir = os.path.abspath(os.path.join(sys.argv[2], 'test'))
# train_dir = os.path.abspath(os.path.join(sys.argv[2], 'train+val'))
work_dir = '/dev/shm/'
test_dir = '/dev/shm/cifar100_test'
train_dir = '/dev/shm/cifar100_train'

# cifar100 = CIFAR100(work_dir, download=False)


def load_file(file_name):
    with open(os.path.join(work_dir, 'cifar-100-python', file_name), 'rb') as meta_f:
#     with open(os.path.join(work_dir, cifar100.base_folder, file_name), 'rb') as meta_f:
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
#for source_file_path, _ in cifar100.test_list:
for source_file_path in ['test']:
    start_idx += unpack_data_file(source_file_path, test_dir, start_idx)

start_idx = 0
# for source_file_path, _ in cifar100.train_list:
for source_file_path in ['train']:
    start_idx += unpack_data_file(source_file_path, train_dir, start_idx)
