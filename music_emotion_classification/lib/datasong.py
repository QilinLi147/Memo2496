import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset



class SONG(Dataset):
    def __init__(self, data_file_path_mel,data_file_path_co, d_labels_path_a, d_labels_path_v, train=True):
        super(SONG, self).__init__()
        # load data
        self.feature_mel = np.load(data_file_path_mel)
        self.feature_co = np.load(data_file_path_co)
        labels_a = np.load(d_labels_path_a)
        labels_v = np.load(d_labels_path_v)
        labels_a[labels_a > 0] = 1
        labels_a[labels_a <= 0] = 0
        labels_v[labels_v > 0] = 1
        labels_v[labels_v <= 0] = 0

        self.labels_a = labels_a.reshape(-1, 1)
        self.labels_v = labels_v.reshape(-1, 1)

        feature_mel = self.feature_mel
        feature_co = self.feature_co
        labels_a = self.labels_a
        labels_v = self.labels_v

        train_size = int(len(feature_mel) * 0.7)  
        test_size = len(feature_mel) - train_size


        train_data_mel = feature_mel[0:train_size, :, :]  
        test_data_mel = feature_mel[train_size:, :, :] 

        train_data_co = feature_co[0:train_size, :, :]  
        test_data_co = feature_co[train_size:, :, :] 
        
        train_data_mel = torch.from_numpy(train_data_mel).to(torch.float32)
        train_data_co = torch.from_numpy(train_data_co).to(torch.float32)
        test_data_mel = torch.from_numpy(test_data_mel).to(torch.float32)
        test_data_co = torch.from_numpy(test_data_co).to(torch.float32)

        labels_a = torch.from_numpy(labels_a).to(torch.float32)
        labels_v = torch.from_numpy(labels_v).to(torch.float32)
        
        train_data_mel = train_data_mel.view([train_size, 128, 87])
        train_data_co = train_data_co.view([train_size, 84, 87])

        test_data_mel = test_data_mel.view([test_size, 128, 87])
        test_data_co = test_data_co.view([test_size, 84, 87])

        train_labels_a = labels_a[0:train_size]
        test_labels_a = labels_a[train_size:]
        train_labels_v = labels_v[0:train_size]
        test_labels_v = labels_v[train_size:]

        if train:
            self.f_data_mel = train_data_mel
            self.f_data_co = train_data_co
            self.f_label_a = train_labels_a
            self.f_label_v = train_labels_v
        else:
            self.f_data_mel = test_data_mel
            self.f_data_co = test_data_co
            self.f_label_a = test_labels_a
            self.f_label_v = test_labels_v

    def __len__(self):
        return len(self.f_label_a)

    def __getitem__(self, index):

        final_data_mel = self.f_data_mel[index]
        final_data_co = self.f_data_co[index]
        final_label_a = self.f_label_a[index]
        final_label_v = self.f_label_v[index]

        return final_data_mel,final_data_co, final_label_a,final_label_v


