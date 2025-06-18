# Dataloader for treepointr v2, based on the PCN dataset.

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

@DATASETS.register_module()
class PCN(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        # Load category and model indexing
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        # Build file list and transformations
        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                },
                {
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }
            ])
        else:
            return data_transforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }
            ])

    def _get_file_list(self, subset):
        file_list = []

        for dc in self.dataset_categories:
            print_log(f'Collecting files of Taxonomy [ID={dc["taxonomy_id"]}, Name={dc["taxonomy_name"]}]', logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                partial_directory = self.partial_points_path % (subset, dc['taxonomy_id'], s)
                partial_paths = [
                    os.path.abspath(os.path.join(partial_directory, p))
                    for p in os.listdir(partial_directory) if p.endswith(".npy")
                ]

                for pp in partial_paths:
                    file_list.append({
                        'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'partial_path': pp,
                        'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    })

        print_log(f'Complete collecting files of the dataset. Total files: {len(file_list)}', logger='PCNDATASET')
        return file_list

    def normalize_pair(self, partial, gt):
        """
        Normalize both GT and partial point clouds using GT's centroid and max-norm.
        """
        centroid = np.mean(gt, axis=0)
        gt_centered = gt - centroid
        scale = np.max(np.linalg.norm(gt_centered, axis=1))

        gt_normalized = gt_centered / scale
        partial_normalized = (partial - centroid) / scale

        return partial_normalized, gt_normalized

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        partial = IO.get(sample['partial_path']).astype(np.float32)
        gt = IO.get(sample['gt_path']).astype(np.float32)

        # Normalize partial & gt using same transformation
        partial, gt = self.normalize_pair(partial, gt)
        data['partial'] = partial
        data['gt'] = gt

        if not np.isfinite(data['partial']).all() or not np.isfinite(data['gt']).all():
            print(f"[WARNING] Invalid values in sample: {sample['partial_path']}")

        assert data['gt'].shape[0] == self.npoints, f"GT does not have expected {self.npoints} points."

        data['file_name'] = sample['partial_path']

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['gt'], data['partial'], data['file_name'])

    def __len__(self):
        return len(self.file_list)
