'''
author: meng-zha
data: 2020/05/26
'''

import pandas as pd
import numpy as np
import os
import h5py

import fire
from tqdm import tqdm

test_split = ["beijing_20140101-20141231"]
trainval_split = [
    "beijing_20150101-20151231",
    "beijing_20160101-20161231",
    "beijing_20170101-20171231",
    "beijing_20180101-20181231",
    "beijing_20190101-20191231",
    "beijing_20200101-20200509"
]
root_path = './air_quality'

def create_database(mode = 'test'):
    if mode == 'test':
        split = test_split
    else:
        split = trainval_split

    # 间断的时间点，time为连续时间的最后一小时所在位置，即2代表0，1，2为连续帧
    breakpoints = []
    test_x = []
    test_y = []

    for folder in split:
        print(folder)
        data_path = os.path.join(root_path,folder)
        for root,dirs,files in os.walk(data_path):
            files.sort()
            for name in tqdm(files):
                if name.find('all')>-1:
                    try:
                        label =  pd.read_csv(os.path.join(root,name))
                        if label.empty or len(label.columns)<35:
                            breakpoints.append(len(test_x))
                            continue
                    except:
                        breakpoints.append(len(test_x))
                        continue
                    label_data = pd.read_csv(os.path.join(root,name)).values[:,3:].astype(float)
                    feat_name = name.replace('all','extra')

                    try:
                        feature =  pd.read_csv(os.path.join(root,feat_name)) 
                        if feature.empty or len(feature.columns)<35:
                            breakpoints.append(len(test_x))
                            continue
                    except:
                        breakpoints.append(len(test_x))
                        continue
                    feat_data = pd.read_csv(os.path.join(root,feat_name)).values[:,3:].astype(float)

                    hours = min(label_data.shape[0]//5,feat_data.shape[0]//8)
                    if hours < 24:
                        breakpoints.append(len(test_x))
                        continue

                    count = 0
                    while(count<hours):
                        test_x.append(np.expand_dims(np.concatenate([count*np.ones((1,35)),feat_data[8*count:8*count+8,:]],axis=0),axis=0))
                        test_y.append(np.expand_dims(label_data[5*count:5*count+5,:],axis=0))
                        count += 1

    test_x = np.concatenate([*test_x],axis=0)
    test_y = np.concatenate([*test_y],axis=0)
    test_x = test_x.transpose(0,2,1)
    test_y = test_y.transpose(0,2,1)
    test_x = np.concatenate([test_x,test_y],axis=2)
    test_f = h5py.File(os.path.join(root_path,f'{mode}_database.h5'),'w')
    test_f.create_dataset('test_x',data=test_x)
    test_f.create_dataset('test_y',data=test_y)
    test_f.close()
    
    np.savetxt(os.path.join(root_path,f'{mode}_break.txt'),np.array(breakpoints),fmt='%d')

def main():
    fire.Fire(create_database)

if __name__ == "__main__":
    main()