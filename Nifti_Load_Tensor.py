#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:31:45 2019

@author: jeg88
"""
import os
import numpy as np
import nibabel as nib
#from nibabel.testing import data_path
from torch.utils.data import Dataset

# load nifti data, transform it to numpy and change the axis order
def nifti_numpy_loader(file_full_dir):
    entry = (nib.load(file_full_dir)).get_fdata().transpose(3, 0, 1, 2) # CDHW
    return entry

# read the traing samples's dir and simulation's dir
def make_dataset(root_path, samples_folder, targets_folder):
    samples = []
    targets = []
    all_samples_name = os.listdir(os.path.join(root_path, samples_folder))
    all_targets_name = os.listdir(os.path.join(root_path, targets_folder))
    for sample in all_samples_name:
        item_sample = os.path.join(root_path, samples_folder, sample)
        samples.append(item_sample)
    for target in all_targets_name:
        item_target = os.path.join(root_path, targets_folder, target)
        targets.append(item_target)    

    return samples, targets      

#samples, targets = make_dataset(root_path, 'training', 'simu_field')


# dataset load: method 2
class RandomSamples(Dataset):
    def __init__(self, root_path, samples_folder, targets_folder, 
                 loader, transform=None):
        super(RandomSamples, self).__init__()
        self.samples, self.targets = make_dataset(root_path, 
                                              samples_folder, targets_folder)
        
        self.loader = loader
        self.transform = transform     
        
    def __getitem__(self, index):
        """
        Args:
            root_path: simu_data/
            samples_folder: training/ or testing/
            targets_folder: label_training/ or label_testing/

        Returns:
            tuple: (sample, target) where target is simulation results.
        """
        sample = self.samples[index]
        target = self.targets[index]
        sample_loader = self.loader(sample)
        target_loader = self.loader(target)
        if self.transform is not None:
            sample_loader = self.transform(sample_loader)
            target_loader = self.transform(target_loader)

        return sample_loader, target_loader

    def __len__(self):
        return len(self.samples) 