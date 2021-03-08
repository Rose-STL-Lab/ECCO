"Functions loading the .pkl version preprocessed data"
from tensorpack import dataflow
from glob import glob
import pickle
import os


class ArgoversePklLoader(dataflow.RNGDataFlow):
    def __init__(self, data_path: str, shuffle: bool=True):
        super(ArgoversePklLoader, self).__init__()
        self.data_path = data_path
        self.shuffle = shuffle
        
    def __iter__(self):
        pkl_list = glob(os.path.join(self.data_path, '*'))
        pkl_list.sort()
        if self.shuffle:
            self.rng.shuffle(pkl_list)
            
        for pkl_path in pkl_list:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            data = {k:v[0] for k, v in data.items()}
            yield data
            
    def __len__(self):
        return len(glob(os.path.join(self.data_path, '*')))
    

def read_pkl_data(data_path: str, batch_size: int, 
                  shuffle: bool=False, repeat: bool=False, **kwargs):
    df = ArgoversePklLoader(data_path=data_path, shuffle=shuffle, **kwargs)
    if repeat:
        df = dataflow.RepeatedData(df, -1)
    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)
    df.reset_state()
    return df

