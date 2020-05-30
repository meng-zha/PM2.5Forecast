'''
author: meng-zha
data: 2020/05/26
'''
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class AirDataset(Dataset):
    def __init__(self,root_path,mode):
        self.mode = mode
        self.root_path = root_path
        with h5py.File(os.path.join(self.root_path,f'{mode}_dataset.h5'),'r') as f:
            self.input = np.array(f['input'])
            self.label = np.array(f['label'])

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self,idx):
        # ['hours' 'SO2' 'NO2' 'O3' 'CO' 'PM2.5' 'AQI'] 
        input,label = self.input[idx],self.label[idx]
        if self.mode == 'train':
            tubs = np.random.normal(0,0.1,35)
            for i in range(35):
                input[:,i,:] += tubs[i]
                label[:,i,:] += tubs[i]
        return input,label