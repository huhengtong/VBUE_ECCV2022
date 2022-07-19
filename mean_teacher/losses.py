# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
import numpy as np


def softmax_mse_loss(input_logits, target_logits, weights):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
#     print(target_softmax)
#     pdb.set_trace()
    num_classes = input_logits.size()[1]
    loss = F.mse_loss(input_softmax, target_softmax, reduction='none')
    loss_sum = torch.sum(loss, dim=1)
    loss_weight = torch.sum(torch.mul(loss_sum, weights)) / num_classes
#     loss_old = F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes
    
    return loss_weight
  
  
def sigmoid_mse_loss(input_logits, target_logits, weights):
    """Takes sigmoid on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = torch.sigmoid(input_logits)
    target_sigmoid = torch.sigmoid(target_logits)

    num_classes = input_logits.size()[1]
    loss = F.mse_loss(input_sigmoid, target_sigmoid, reduction='none')
    loss_sum = torch.sum(loss, dim=1)
    loss_weight = torch.sum(torch.mul(loss_sum, weights)) / num_classes
#     loss_old = F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes
    
    return loss_weight


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

  
class PartSumCrossEntropyLoss(torch.nn.Module):
    r"""This function sums up the predicted propobilities of all the
        classes in the target set and calculates the log-value.
    Args:
        - Input: tensor of size (N, C) where C = number of classes, or
          of size (N, C, d_1, d_2, ..., d_K) with K >= 1 in the case of
          K-dimensional loss.
        - Target: tensor of size (N, C) where targets[i] = 0, 1 indicates
          whether this class is in the label set, or
          of size (N, C, d_1, d_2, ..., d_K) with K >= 1 in the case of
          K-dimensional loss.
        - Output: scalar. (N, d_1, d_2, ..., d_K) with K >= 1 in the case
          of K-dimensional loss.
    """ 
    def __init__(self):
        super(PartSumCrossEntropyLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, input, target):
        input = self.softmax(input)
        loss = -1 * torch.log(1e-12 + (input * target.float()).sum(-1))
        return loss.mean()
  

class PartMeanCrossEntropyLoss(torch.nn.Module):
    r"""This function considers propobilities equal.
    Args:
        - Input: tensor of size (N, C) where C = number of classes, or
          of size (N, C, d_1, d_2, ..., d_K) with K >= 1 in the case of
          K-dimensional loss.
        - Target: tensor of size (N, C) where targets[i] = 0, 1 indicates
          whether this class is in the label set, or
          of size (N, C, d_1, d_2, ..., d_K) with K >= 1 in the case of
          K-dimensional loss.
        - Output: scalar. (N, d_1, d_2, ..., d_K) with K >= 1 in the case
          of K-dimensional loss.
    """ 
    def __init__(self):
        super(PartMeanCrossEntropyLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, target):
        log_input = self.log_softmax(input)
        indexs = (torch.sum(target, dim=1) > 0).nonzero().squeeze()       
        loss = -1 * (log_input[indexs] * target[indexs].float()).sum(-1) / (target[indexs].sum(1).float())
        return loss.mean()
    