#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from datasets.helper import get_lane_direction
# from tensorpack import dataflow
import time
import gc
import pickle
import helper
import time
import glob
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

dataset_path = '/path/to/dataset/argoverse_forecasting/'

val_path = os.path.join(dataset_path, 'val', 'data')
train_path = os.path.join(dataset_path, 'train', 'data')

class ArgoverseTest(object):
    """
    Data flow for argoverse dataset
    """

    def __init__(self, file_path: str, shuffle: bool = True, random_rotation: bool = False,
                 max_car_num: int = 50, freq: int = 10, use_interpolate: bool = False, 
                 use_lane: bool = False, use_mask: bool = True):
        if not os.path.exists(file_path):
            raise Exception("Path does not exist.")

        self.afl = ArgoverseForecastingLoader(file_path)
        self.shuffle = shuffle
        self.random_rotation = random_rotation
        self.max_car_num = max_car_num
        self.freq = freq
        self.use_interpolate = use_interpolate
        self.am = ArgoverseMap()
        self.use_mask = use_mask
        self.file_path = file_path
        

    def get_feat(self, scene):

        data, city = self.afl[scene].seq_df, self.afl[scene].city

        lane = np.array([[0., 0.]], dtype=np.float32)
        lane_drct = np.array([[0., 0.]], dtype=np.float32)


        tstmps = data.TIMESTAMP.unique()
        tstmps.sort()

        data = self._filter_imcomplete_data(data, tstmps, 50)

        data = self._calc_vel(data, self.freq)

        agent = data[data['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].values[0]

        car_mask = np.zeros((self.max_car_num, 1), dtype=np.float32)
        car_mask[:len(data.TRACK_ID.unique())] = 1.0

        feat_dict = {'city': city, 
                     'lane': lane, 
                     'lane_norm': lane_drct, 
                     'scene_idx': scene,  
                     'agent_id': agent, 
                     'car_mask': car_mask}

        pos_enc = [subdf[['X', 'Y']].values[np.newaxis,:] 
                   for _, subdf in data[data['TIMESTAMP'].isin(tstmps[:19])].groupby('TRACK_ID')]
        pos_enc = np.concatenate(pos_enc, axis=0)
        # pos_enc = self._expand_dim(pos_enc)
        feat_dict['pos_2s'] = self._expand_particle(pos_enc, self.max_car_num, 0)

        vel_enc = [subdf[['vel_x', 'vel_y']].values[np.newaxis,:] 
                   for _, subdf in data[data['TIMESTAMP'].isin(tstmps[:19])].groupby('TRACK_ID')]
        vel_enc = np.concatenate(vel_enc, axis=0)
        # vel_enc = self._expand_dim(vel_enc)
        feat_dict['vel_2s'] = self._expand_particle(vel_enc, self.max_car_num, 0)

        pos = data[data['TIMESTAMP'] == tstmps[19]][['X', 'Y']].values
        # pos = self._expand_dim(pos)
        feat_dict['pos0'] = self._expand_particle(pos, self.max_car_num, 0)
        vel = data[data['TIMESTAMP'] == tstmps[19]][['vel_x', 'vel_y']].values
        # vel = self._expand_dim(vel)
        feat_dict['vel0'] = self._expand_particle(vel, self.max_car_num, 0)
        track_id =  data[data['TIMESTAMP'] == tstmps[19]]['TRACK_ID'].values
        feat_dict['track_id0'] = self._expand_particle(track_id, self.max_car_num, 0, 'str')
        feat_dict['frame_id0'] = 0
        
        for t in range(31):
            pos = data[data['TIMESTAMP'] == tstmps[19 + t]][['X', 'Y']].values
            # pos = self._expand_dim(pos)
            feat_dict['pos' + str(t)] = self._expand_particle(pos, self.max_car_num, 0)
            vel = data[data['TIMESTAMP'] == tstmps[19 + t]][['vel_x', 'vel_y']].values
            # vel = self._expand_dim(vel)
            feat_dict['vel' + str(t)] = self._expand_particle(vel, self.max_car_num, 0)
            track_id =  data[data['TIMESTAMP'] == tstmps[19 + t]]['TRACK_ID'].values
            feat_dict['track_id' + str(t)] = self._expand_particle(track_id, self.max_car_num, 0, 'str')
            feat_dict['frame_id' + str(t)] = t

        return feat_dict
    
    def __len__(self):
        return len(glob.glob(os.path.join(self.file_path, '*')))

    @classmethod
    def _expand_df(cls, data, city_name):
        timestps = data['TIMESTAMP'].unique().tolist()
        ids = data['TRACK_ID'].unique().tolist()
        df = pd.DataFrame({'TIMESTAMP': timestps * len(ids)}).sort_values('TIMESTAMP')
        df['TRACK_ID'] = ids * len(timestps)
        df['CITY_NAME'] = city_name
        return pd.merge(data, df, on=['TIMESTAMP', 'TRACK_ID'], how='right')


    @classmethod
    def __calc_vel_generator(cls, df, freq=10):
        for idx, subdf in df.groupby('TRACK_ID'):
            sub_df = subdf.copy().sort_values('TIMESTAMP')
            sub_df[['vel_x', 'vel_y']] = sub_df[['X', 'Y']].diff() * freq
            yield sub_df.iloc[1:, :]

    @classmethod
    def _calc_vel(cls, df, freq=10):
        return pd.concat(cls.__calc_vel_generator(df, freq=freq), axis=0)
    
    @classmethod
    def _expand_dim(cls, ndarr, dtype=np.float32):
        return np.insert(ndarr, 2, values=0, axis=-1).astype(dtype)
    
    @classmethod
    def _linear_interpolate_generator(cls, data, col=['X', 'Y']):
        for idx, df in data.groupby('TRACK_ID'):
            sub_df = df.copy().sort_values('TIMESTAMP')
            sub_df[col] = sub_df[col].interpolate(limit_direction='both')
            yield sub_df.ffill().bfill()
    
    @classmethod
    def _linear_interpolate(cls, data, col=['X', 'Y']):
        return pd.concat(cls._linear_interpolate_generator(data, col), axis=0)
    
    @classmethod
    def _filter_imcomplete_data(cls, data, tstmps, window=20):
        complete_id = list()
        for idx, subdf in data[data['TIMESTAMP'].isin(tstmps[:window])].groupby('TRACK_ID'):
            if len(subdf) == window:
                complete_id.append(idx)
        return data[data['TRACK_ID'].isin(complete_id)]
    
    @classmethod
    def _expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)
    
    
class process_utils(object):
            
    @classmethod
    def expand_dim(cls, ndarr, dtype=np.float32):
        return np.insert(ndarr, 2, values=0, axis=-1).astype(dtype)
    
    @classmethod
    def expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)
    

def get_max_min(datas):
    mask = datas['car_mask']
    slicer = mask[0].astype(bool).flatten()
    pos_keys = ['pos0'] + ['pos_2s']
    max_x = np.concatenate([np.max(np.stack(datas[pk])[0,slicer,...,0]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    min_x = np.concatenate([np.min(np.stack(datas[pk])[0,slicer,...,0]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    max_y = np.concatenate([np.max(np.stack(datas[pk])[0,slicer,...,1]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    min_y = np.concatenate([np.min(np.stack(datas[pk])[0,slicer,...,1]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    max_x = np.max(max_x, axis=-1) + 10
    max_y = np.max(max_y, axis=-1) + 10
    min_x = np.max(min_x, axis=-1) - 10
    min_y = np.max(min_y, axis=-1) - 10
    return min_x, max_x, min_y, max_y


def process_func(putil, datas, am):
    
    city = datas['city'][0]
    x_min, x_max, y_min, y_max = get_max_min(datas)

    seq_lane_props = am.city_lane_centerlines_dict[city]

    lane_centerlines = []
    lane_directions = []

    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_centerlines.append(lane_cl[1:])
            lane_drct = np.diff(lane_cl, axis=0)
            lane_directions.append(lane_drct)
    if len(lane_centerlines) > 0:
        lane = np.concatenate(lane_centerlines, axis=0)
        # lane = putil.expand_dim(lane)
        lane_drct = np.concatenate(lane_directions, axis=0)
        # lane_drct = putil.expand_dim(lane_drct)[...,:3]

        datas['lane'] = [lane]
        datas['lane_norm'] = [lane_drct]
        return datas
    else:
        return datas
    
    
if __name__ == '__main__':
    am = ArgoverseMap()
    putil = process_utils()

    afl_train = ArgoverseForecastingLoader(os.path.join(dataset_path, 'train', 'data'))
    afl_val = ArgoverseForecastingLoader(os.path.join(dataset_path, 'val', 'data'))
    at_train = ArgoverseTest(os.path.join(dataset_path, 'train', 'data'), max_car_num=60)
    at_val = ArgoverseTest(os.path.join(dataset_path, 'val', 'data'), max_car_num=60)
    
    
    print("++++++++++++++++++++ START TRAIN ++++++++++++++++++++")
    train_num = len(afl_train)
    batch_start = time.time()
    os.mkdir(os.path.join(dataset_path, 'train/lane_data'))
    for i, scene in enumerate(range(train_num)):
        if i % 1000 == 0:
            batch_end = time.time()
            print("SAVED ============= {} / {} ....... {}".format(i, train_num, batch_end - batch_start))
            batch_start = time.time()

        data = {k:[v] for k, v in at_train.get_feat(scene).items()}
        datas = process_func(putil, data, am)
        with open(os.path.join(dataset_path, 'train/lane_data', str(datas['scene_idx'][0])+'.pkl'), 'wb') as f:
            pickle.dump(datas, f)
    
    print("++++++++++++++++++++ START VAL ++++++++++++++++++++")
    val_num = len(afl_val)
    batch_start = time.time()
    os.mkdir(os.path.join(dataset_path, 'val/lane_data'))
    for i, scene in enumerate(range(val_num)):
        if i % 1000 == 0:
            batch_end = time.time() 
            print("SAVED ============= {} / {} ....... {}".format(i, val_num, batch_end - batch_start))
            batch_start = time.time()

        data = {k:[v] for k, v in at_val.get_feat(scene).items()}
        datas = process_func(putil, data, am)
        with open(os.path.join(dataset_path, 'val/lane_data', str(datas['scene_idx'][0])+'.pkl'), 'wb') as f:
            pickle.dump(datas, f)


    