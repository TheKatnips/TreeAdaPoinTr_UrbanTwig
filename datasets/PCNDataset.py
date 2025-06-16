# Dadaloader for treepointr v2, based on the PCN dataset.

import torch.utils.data as data
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import data_transforms
from .io import IO
import json
from .build import DATASETS
from utils.logger import *

# Reference implementation inspired from:
# https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCN(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        #self.cars = config.CARS

        # Load category and model indexing
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                # Optionally filter only cars
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        # Build file list and transformations
        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        # Only apply ToTensor; no data augmentation or upsampling
        return data_transforms.Compose([{
            'callback': 'ToTensor',
            'objects': ['partial', 'gt']
        }])

    def _get_file_list(self, subset):
        """
        Construct the list of data samples based on the config and category file.
        Each partial .npy file is treated as an independent input, paired with a single GT.
        """
        file_list = []

        for dc in self.dataset_categories:
            print_log(f'Collecting files of Taxonomy [ID={dc["taxonomy_id"]}, Name={dc["taxonomy_name"]}]', logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                # Directory containing multiple partial views for one model
                partial_directory = self.partial_points_path % (subset, dc['taxonomy_id'], s)

                # Collect all partial .npy file paths (multiple per model)
                partial_paths = [
                    os.path.abspath(os.path.join(partial_directory, p))
                    for p in os.listdir(partial_directory) if p.endswith(".npy")
                ]

                # For each partial file, store the corresponding GT path
                for pp in partial_paths:
                    file_list.append({
                        'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'partial_path': pp,
                        'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    })

        print_log(f'Complete collecting files of the dataset. Total files: {len(file_list)}', logger='PCNDATASET')
        return file_list

    def pc_norm(self, pc):
        """
        Normalize the point cloud to zero-mean and unit-max-norm.
        Args:
            pc (np.ndarray): Input point cloud (N x 3 or N x C)
        Returns:
            np.ndarray: Normalized point cloud
        """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        """
        Load a sample from the dataset: a normalized partial and GT point cloud pair.
        """
        sample = self.file_list[idx]
        data = {}

        # Load partial and GT point clouds
        partial = IO.get(sample['partial_path']).astype(np.float32)
        gt = IO.get(sample['gt_path']).astype(np.float32)

        # Normalize both point clouds
        data['partial'] = self.pc_norm(partial)
        data['gt'] = self.pc_norm(gt)

        # Sanity check for invalid data
        if not np.isfinite(data['partial']).all() or not np.isfinite(data['gt']).all():
            print(f"[WARNING] Invalid values in sample: {data['file_name']}")

        # Ensure ground truth has the expected number of points
        assert data['gt'].shape[0] == self.npoints

        data['file_name'] = sample['partial_path']

        # Apply transforms (currently only ToTensor)
        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['gt'], data['partial'], data['file_name'])

    def __len__(self):
        return len(self.file_list)
