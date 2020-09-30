#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:17:04 2019

@author: jeg88
"""

import os
import re
import collections

#from medicaltorch import transforms as mt_transforms

from tqdm import tqdm
import numpy as np
import nibabel as nib
#from nibabel.testing import data_path
from torch.utils.data import Dataset
#from medicaltorch import datasets as mt_datasets
from Nifti_Load_Tensor import nifti_numpy_loader, make_dataset

class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def keys(self):
        return self.metadata.keys()

class SimulationPair3D(object):
    """This class is used to build 3D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32).transpose(3, 0, 1, 2) # CDHW

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32).transpose(3, 0, 1, 2) # CDHW

        return input_data, gt_data

class SimuNibsDataset(Dataset):
    """This is a generic class for 3D segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, cache=True,
                 transform=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.cache = cache
        self.canonical = canonical

        self._load_filenames()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SimulationPair3D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def __len__(self):
        """Return the dataset size."""
        return len(self.handlers)

    def __getitem__(self, index):
        """Return the specific index pair volume (input, ground truth).

        :param index: volume index.
        """
        input_img, gt_img = self.handlers[index].get_pair_data()
        data_dict = {
            'input': input_img,
            'gt': gt_img
        }
        if self.transform is not None:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
            data_dict = {
                'input': input_img,
                'gt': gt_img
            }
        return data_dict


class SimuNibsTrain(SimuNibsDataset):
    """This is the Simulation dataset from SimNIBS

    :param root_dir: the directory containing the training dataset.
    :param site_ids: a list of site ids to filter (i.e. [1, 3]).
    :param subj_ids: the list of subject ids to filter.
    :param rater_ids: the list of the rater ids to filter.
    :param transform: the transformations that should be applied.
    :param cache: if the data should be cached in memory or not.
    :param slice_axis: axis to make the slicing (default axial).
    """
    def __init__(self, root_dir, samples_folder, targets_folder, 
                 cache=True, transform=None, canonical=False):

        self.root_dir = root_dir
        self.filename_pairs = []
        
        self.samples, self.targets = make_dataset(self.root_dir, 
                                              samples_folder, targets_folder)
        for idx in range(len(self.samples)):
            input_filename = self.samples[idx]
            gt_filename = self.targets[idx]
            self.filename_pairs.append((input_filename, gt_filename))
     
        super().__init__(self.filename_pairs, cache,
                         transform,  canonical)
