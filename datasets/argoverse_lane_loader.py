"Functions loading the .pkl version preprocessed data"
from tensorpack import dataflow
from glob import glob
import pickle
import os
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap


class ArgoversePklLoader(dataflow.RNGDataFlow):
    def __init__(self, data_path: str, shuffle: bool=True, max_lane_nodes=650, min_lane_nodes=0):
        super(ArgoversePklLoader, self).__init__()
        self.data_path = data_path
        self.shuffle = shuffle
        self.max_lane_nodes = max_lane_nodes
        self.min_lane_nodes = min_lane_nodes
        self.am = ArgoverseMap()
        
    def __iter__(self):
        pkl_list = glob(os.path.join(self.data_path, '*'))
        pkl_list.sort()
        np.random.RandomState(233).shuffle(pkl_list)
        if self.shuffle:
            self.rng.shuffle(pkl_list)
            
        for pkl_path in pkl_list:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            data = {k:v[0] for k, v in data.items()}
            
            # displacement = np.linalg.norm(data['pos_2s'][:,-1] - data['pos_2s'][:,0], axis=-1)
            # displace_mask = displacement < self.displace_filter
            # da_mask = self.am.get_raster_layer_points_boolean(data['pos_2s'].reshape(-1, 3), data['city'], "driveable_area")
            # da_mask = da_mask.reshape(*data['pos_2s'].shape[:-1])
            # da_mask = da_mask.all(-1)
            # mask = da_mask[data['track_id0'] == data['agent_id']] = True
            # convert_keys = (['pos' + str(i) for i in range(31)] + 
            #                 ['vel' + str(i) for i in range(31)] + 
            #                 ['pos_2s', 'vel_2s', 'car_mask'])
            # for k in convert_keys:
            #     data[k][~mask] = 0
            
            # norm = np.linalg.norm(data['lane_norm'], axis=-1)
            # data['lane'] = data['lane'][norm > 0]
            # data['lane_norm'] = data['lane_norm'][norm > 0] / norm[norm > 0, np.newaxis] 
            lane_mask = np.zeros(self.max_lane_nodes, dtype=np.float32)
            lane_mask[:len(data['lane'])] = 1.0
            data['lane_mask'] = lane_mask
            
            # xy = data['pos_2s']
            # agent = data['track_id0'] == data['agent_id']

            # distance_2 = np.sum(np.square(xy - data['pos0'][agent, np.newaxis]), axis=2)
            # mask = np.nanmin(distance_2, axis=1) > 40**2

            # convert_keys = (['pos' + str(i) for i in range(31)] + 
            #                 ['vel' + str(i) for i in range(31)] + 
            #                 ['pos_2s', 'vel_2s', 'car_mask'])
            # for key in convert_keys:
            #     data[key][mask] = 0
            
            if data['lane'].shape[0] > self.max_lane_nodes:
                continue
                
            if data['lane'][0].shape[0] < self.min_lane_nodes:
                continue
            
            data['lane'] = self.expand_particle(data['lane'], self.max_lane_nodes, 0)
            data['lane_norm'] = self.expand_particle(data['lane_norm'], self.max_lane_nodes, 0)
            
            yield data
            
    def __len__(self):
        return len(glob(os.path.join(self.data_path, '*')))
    
    @classmethod
    def expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)
    

def read_pkl_data(data_path: str, batch_size: int, 
                  shuffle: bool=False, repeat: bool=False, **kwargs):
    df = ArgoversePklLoader(data_path=data_path, shuffle=shuffle, **kwargs)
    if repeat:
        df = dataflow.RepeatedData(df, -1)
    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)
    df.reset_state()
    return df


