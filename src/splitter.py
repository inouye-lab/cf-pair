from re import L
import torch
import copy
import numpy as np
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import ImageFolder
from wilds.datasets.wilds_dataset import WILDSSubset, WILDSDataset


class RandomSplitter(object):
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
    
    def split(self, dataset, attributes, num, transform=None):
        if isinstance(dataset, WILDSSubset):
            indices = dataset.indices
            ori_dataset = dataset.dataset
        elif isinstance(dataset, WILDSDataset):
            indices = np.arange(len(dataset))
            ori_dataset = dataset
        
        domain_field = dataset._metadata_fields.index(attributes)
        all_id = np.unique(ori_dataset._metadata_array[indices, domain_field])
        perm_id = self.rng.permutation(all_id)
        sub_id = perm_id[0:num]
        mask = torch.zeros(len(indices))
        for i in sub_id:
            mask = torch.logical_or(mask, ori_dataset._metadata_array[indices, domain_field]==i)
        
        
        return WILDSSubset(ori_dataset, indices[mask].tolist(), transform=transform), WILDSSubset(ori_dataset, indices[torch.logical_not(mask)].tolist(), transform=transform)
