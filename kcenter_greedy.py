"""Returns points that minimizes the maximum distance of any point to a center.
Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).
Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import abc
import pdb
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from sklearn.externals import joblib
#from sampling_methods.sampling_def import SamplingMethod


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
        return flat_X


    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):

    def __init__(self, features, metric='euclidean'):
#         self.X = X
#         self.y = y
#         self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = features.cuda()
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.features.shape[0]
        self.already_selected = []
        
#         joblib.dump(self.features, '/cache/features.npy')
#         self.features = joblib.load('/cache/features.npy', mmap_mode='r+')

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if len(cluster_centers) > 0:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
#             dist_list = []
#             dist_array = np.zeros((len(self.features), len(x))).astype(np.float32)
#             for i in range(len(x)):
#                 dist = pairwise_distances(self.features, x[i].reshape(1, -1), metric=self.metric)
#                 dist_array[:,i] = np.squeeze(dist).astype(np.float32)
#                 dist_list.append(dist)
#             dist = np.stack(dist_list, axis=0)
#             dist = pairwise_distances(self.features, x, metric=self.metric, n_jobs=-1)
            if len(x.size()) == 1:
                x = torch.unsqueeze(x, dim=0)
            dist = torch.cdist(self.features, x, p=2)

        if self.min_distances is None:
#             self.min_distances,_ = torch.min(dist, dim=1)
            self.min_distances = np.amin(dist.cpu().numpy(), axis=1).reshape(-1,1)
        else:
#             self.min_distances = torch.minimum(self.min_distances, dist)
            self.min_distances = np.minimum(self.min_distances, dist.cpu().numpy())
            del dist

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

#         try:
#             # Assumes that the transform function takes in original data and not
#             # flattened data.
#             print('Getting transformed features...')
#             self.features = model.transform(self.X)
#             print('Calculating distances...')
#             self.update_distances(already_selected, only_new=False, reset_dist=True)
#         except:
        print('Using flat_X as features.')
        self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for i in range(N):
            if i % 1000 == 0:
                print('************************', i)
#             if self.already_selected is None:
#                 # Initialize centers with a randomly selected datapoint
#                 ind = np.random.choice(np.arange(self.n_obs))
#             else:  
            ind = np.argmax(self.min_distances)
            
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))
        self.already_selected = already_selected
        return new_batch
    
    
    
    
    