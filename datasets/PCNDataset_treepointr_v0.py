import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset) # , self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints', # RandomSamplePoints
                'parameters': {
                    'n_points': 2048  # replace with 8192?  # 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            # },{
                # 'callback': 'NormalizeObjectPose',
                # 'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints', # RandomSamplePoints
                'parameters': {
                    'n_points': 2048 # replace with 8192? # original: 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset):  #, n_renderings=18
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            # for s in samples:
                # file_list.append({
                    # 'taxonomy_id':
                    # dc['taxonomy_id'],
                    # 'model_id': s,
                    # 'partial_path':
                        #[self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        #for i in range(n_renderings)],
                        # os.listdir(self.partial_points_path % (subset, dc['taxonomy_id'], s)),
                    # 'gt_path':
                    # self.complete_points_path % (subset, dc['taxonomy_id'], s),
                # })
            for s in samples:
                partial_directory = self.partial_points_path % (subset, dc['taxonomy_id'], s)
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path':
                        #[self.partial_points_path % (subset, dc['taxonomy_id'], s, i) 
                        #for i in range(len(os.listdir(partial_directory))], 
                        [os.path.abspath(os.path.join(partial_directory, p)) for p in os.listdir(partial_directory) if p.endswith(".npy")],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        #print('file list from _get_file_list: ' + file_list)
        return file_list

    # normalizes the point cloud by centering it around its centroid and scaling it such that the maximum distance from the centroid is 1.
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        #rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0
        #print(rand_idx)

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            #print(file_path)
            if type(file_path) == list:
                rand_idx = random.randint(0, len(file_path) - 1) if self.subset=='train' else 0
                file_path = file_path[rand_idx]
            #print(file_path)
            data_object =  IO.get(file_path).astype(np.float32) # data[ri] = IO.get(file_path) 
            data[ri] = self.pc_norm(data_object) 
    

        assert data['gt'].shape[0] == self.npoints
        data['file_name'] = file_path # ???
        #print(data['file_name'])

        if self.transforms is not None:
            data = self.transforms(data)

        #return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], data['file_name']) # ???

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class PCNv2(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 18 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)
    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
       