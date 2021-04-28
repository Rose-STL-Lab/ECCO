"Functions loading the .pkl version preprocessed data"
from glob import glob
import pickle
import os
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
from torch.utils.data import IterableDataset, DataLoader


class ArgoverseDataset(IterableDataset):
    def __init__(self, data_path: str, transform=None, 
                 max_lane_nodes=650, min_lane_nodes=0, shuffle=True):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        if shuffle:
            np.random.shuffle(self.pkl_list)
        else:
            self.pkl_list.sort()
        self.max_lane_nodes = max_lane_nodes
        self.min_lane_nodes = min_lane_nodes
        
    def __len__(self):
        return len(self.pkl_list)
    
    def __iter__(self):
        # pkl_path = self.pkl_list[idx]
        for pkl_path in self.pkl_list:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            # data = {k:v[0] for k, v in data.items()}
            lane_mask = np.zeros(self.max_lane_nodes, dtype=np.float32)
            lane_mask[:len(data['lane'][0])] = 1.0
            data['lane_mask'] = [lane_mask]

            if data['lane'][0].shape[0] > self.max_lane_nodes:
                continue

            if data['lane'][0].shape[0] < self.min_lane_nodes:
                continue

            data['lane'] = [self.expand_particle(data['lane'][0], self.max_lane_nodes, 0)]
            data['lane_norm'] = [self.expand_particle(data['lane_norm'][0], self.max_lane_nodes, 0)]

            if self.transform:
                data = self.transform(data)

            yield data
    
    @classmethod
    def expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)
    
    
def cat_key(data, key):
    result = []
    for d in data:
        result = result + d[key]
    return result


def dict_collate_func(data):
    keys = data[0].keys()
    data = {key: cat_key(data, key) for key in keys}
    return data


def read_pkl_data(data_path: str, batch_size: int, 
                  shuffle: bool=False, repeat: bool=False, **kwargs):
    dataset = ArgoverseDataset(data_path=data_path, shuffle=shuffle, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dict_collate_func)
    if repeat:
        while True:
            for data in loader:
                yield data
    else:
        for data in loader:
            yield data
            
