'''
author: meng-zha
data: 2020/05/27
'''
import h5py
import numpy as np
import pandas as pd
import os
import fire
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm


class Cleaner(object):
    def __init__(self,
                 root_path='./air_quality',
                 seed=None,
                 past=24,
                 forward=6):
        self.root_path = root_path
        self.seed = seed
        self.past = past
        self.foward = forward

        self.trainval_path = os.path.join(root_path, 'trainval_database.h5')
        self.test_path = os.path.join(root_path, 'test_database.h5')

        self.loaddata()

        self.trainval_break = np.unique(
            np.loadtxt(os.path.join(root_path,
                                    'trainval_break.txt')).astype(int))
        self.trainval_break += 1
        self.trainval_break = np.concatenate(
            [[0], self.trainval_break, [self.trainval.shape[0]]], axis=0)
        self.test_break = np.unique(
            np.loadtxt(os.path.join(root_path, 'test_break.txt')).astype(int))
        self.test_break += 1
        self.test_break = np.concatenate(
            [[0], self.test_break, [self.test.shape[0]]], axis=0)

    def loaddata(self):
        with h5py.File(self.trainval_path, 'r') as f:
            self.trainval = np.array(f['test_x']).astype(np.float)
            self.trainval_y = np.array(f['test_y']).astype(np.float)
        with h5py.File(self.test_path, 'r') as f:
            self.test = np.array(f['test_x']).astype(np.float)
            self.test_y = np.array(f['test_y']).astype(np.float)

    def statics(self, mode='trainval'):
        if mode == 'trainval':
            feature_size = self.trainval.shape[2]
            total = self.trainval[..., 0].size
            features = self.trainval
        else:
            feature_size = self.test.shape[2]
            total = self.test[..., 0].size
            features = self.test

        for i in range(feature_size):
            name = self.num2names(i)
            feature = features[..., i]
            plt.clf()
            plt.hist(feature[~np.isnan(feature)], 500)
            nan_rate = (np.isnan(feature)).sum() / total
            plt.title(f'{name}:nan_rate = {nan_rate}')
            plt.savefig(f'./figures/{mode}_{name}.png')

    def norm(self):
        # time cycle -> sin(time)
        self.trainval[..., 0] = np.sin(self.trainval[..., 0] * 12 / np.pi)
        self.test[..., 0] = np.sin(self.test[..., 0] * 12 / np.pi)

        abandoned = []
        mean_list = []
        var_list = []
        feature_size = self.trainval.shape[2]
        for i in range(1, feature_size):
            feature = self.trainval[..., i]
            feature_test = self.test[..., i]
            nan_rate = (np.isnan(feature)).sum() / feature.size
            if nan_rate > 0.2:
                # too many missed, then abandon feaure
                abandoned.append(i)
            else:
                # long tail -> norm
                feature[~np.isnan(feature)] = np.log(
                    feature[~np.isnan(feature)] + 1)
                feature_test[~np.isnan(feature_test)] = np.log(
                    feature_test[~np.isnan(feature_test)] + 1)

                import pdb; pdb.set_trace()
                # norm[0,1]
                mean = feature[~np.isnan(feature)].mean()
                mean_list.append(mean)
                var = feature[~np.isnan(feature)].std()
                var_list.append(var)
                feature[~np.isnan(feature)] = (feature[~np.isnan(feature)] -
                                               mean) / var
                feature_test[~np.isnan(feature_test)] = (
                    feature_test[~np.isnan(feature_test)] - mean) / var

                self.trainval[..., i] = feature
                self.test[..., i] = feature_test

        self.names = []
        for i in range(self.trainval.shape[2]):
            self.names.append(self.num2names(i))
        self.names = np.delete(np.array(self.names), abandoned, axis=0)
        print(self.names)
        para = np.vstack([self.names[1:], mean_list, var_list])
        para = pd.DataFrame(para)
        para.to_csv(os.path.join(self.root_path, 'mean_var.csv'),
                    header=False,
                    index=False)

        self.trainval = np.delete(self.trainval, abandoned, axis=2)
        self.test = np.delete(self.test, abandoned, axis=2)

        return mean_list

    def clean(self, mean_list, mode):
        ''' move the nan value '''
        if mode == 'trainval':
            features = self.trainval
            mode_break = self.trainval_break
        else:
            features = self.test
            mode_break = self.test_break

        index = np.where(np.isnan(features))

        # init the nan using the 0 mean value
        features[np.isnan(features)] = 0

        for i, ob, feat in (zip(*index)):
            prev = i - 10
            nex = i + 10
            division = np.where(mode_break <= i)[0][-1]
            if prev < mode_break[division]:
                prev = mode_break[division]
            if nex >= mode_break[division + 1]:
                nex = mode_break[division + 1] - 1

            x = list(range(prev, nex + 1))
            x.remove(i)
            y = []
            if len(x) <= 5:
                # divison too short < 5hours
                continue
            for j in x:
                y.append(features[j, ob, feat])
            tck = interpolate.splrep(x, y)
            y_bspline = interpolate.splev(i, tck)
            features[i, ob, feat] = float(y_bspline)
        

    def create_dataset(self):
        mean_list = self.norm()
        self.clean(mean_list, 'trainval')
        self.clean(mean_list, 'test')

        # get input as (24*features,6*label)
        # trainval
        train_data = {'data': [], 'label': []}
        val_data = {'data': [], 'label': []}
        test_data = {'data': [], 'label': []}

        for i in tqdm(range(self.trainval.shape[0])):
            division = np.where(self.trainval_break <= i)[0][-1]
            if i + self.past + self.foward >= self.trainval_break[division +
                                                                  1]:
                continue
            input_data = np.expand_dims(self.trainval[i:i + self.past], axis=0)
            label_data = np.expand_dims(self.trainval[i + self.past:i +
                                                      self.past + self.foward],
                                        axis=0)
            if self.seed is None:
                rand = np.random.uniform()
            else:
                np.random.seed(self.seed + i)
                rand = np.random.uniform()
            if rand < 0.2:
                # eval
                val_data['data'].append(input_data)
                val_data['label'].append(label_data)
            else:
                # train
                train_data['data'].append(input_data)
                train_data['label'].append(label_data)

        # test data
        for i in tqdm(range(self.test.shape[0])):
            division = np.where(self.test_break <= i)[0][-1]
            if i + self.past + self.foward >= self.test_break[division + 1]:
                continue
            input_data = np.expand_dims(self.test[i:i + self.past], axis=0)
            label_data = np.expand_dims(self.test[i + self.past:i + self.past +
                                                  self.foward],
                                        axis=0)
            # test
            test_data['data'].append(input_data)
            test_data['label'].append(label_data)

        train_data['data'] = np.concatenate([*train_data['data']], axis=0)
        train_data['label'] = np.concatenate([*train_data['label']], axis=0)
        val_data['data'] = np.concatenate([*val_data['data']], axis=0)
        val_data['label'] = np.concatenate([*val_data['label']], axis=0)
        test_data['data'] = np.concatenate([*test_data['data']], axis=0)
        test_data['label'] = np.concatenate([*test_data['label']], axis=0)

        with h5py.File(os.path.join(self.root_path, 'train_dataset.h5'),
                       'w') as f:
            # 33451,24,35,7
            f.create_dataset('input', data=train_data['data'])
            f.create_dataset('label', data=train_data['label'])
        with h5py.File(os.path.join(self.root_path, 'val_dataset.h5'),
                       'w') as f:
            # 8178,24,35,7
            f.create_dataset('input', data=val_data['data'])
            f.create_dataset('label', data=val_data['label'])
        with h5py.File(os.path.join(self.root_path, 'test_dataset.h5'),
                       'w') as f:
            # 5678,24,35,7
            f.create_dataset('input', data=test_data['data'])
            f.create_dataset('label', data=test_data['label'])

    def num2names(self, num):
        names = [
            'hours', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'CO',
            'CO_24h', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'AQI'
        ]
        return names[num]


if __name__ == "__main__":
    fire.Fire(Cleaner)
