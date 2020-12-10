from torch.utils import data
from transforms import get_transform
from utils import seed_everything, to_onehot, label_smoothing
from datasplit import split
import torch
import numpy as np
import os
import cv2

seed_everything(42)

num_classes = 3

def get_datas():
    """
    x : list of img file path
    y : list of label(integer)
    """
    
    data = split["train"]
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []

    for i,k in enumerate(data):
        i = [0,1,2,2,2,2][i]
        
        if k !='bald':
            for j,person in enumerate(data[k]):
                base_dir = '../full_data/new_updated/{}/{}/'.format(k,person)
                files = os.listdir(base_dir)
                files = [base_dir + file for file in files if file.endswith('.jpg')]
                if j<5:
                    train_x +=files
                    train_y += [i]*len(files)
                else:
                    valid_x +=files
                    valid_y += [i]*len(files)
        else:
            base_dir = '../full_data/new_updated/bald/Bald_train/'
            files = os.listdir(base_dir)
            files = [base_dir + file for file in files if file.endswith('.jpg')]
            train_x += files[:1900]
            valid_x += files[1900:]
            train_y += [i] * len(files[:1900])
            valid_y += [i] * len(files[1900:])
    
    return train_x,train_y,valid_x,valid_y

def get_test_datas():
    data = split["test"]
    x = []
    y = []

    for i,k in enumerate(data):
        i = [0,1,2,2,2,2][i]
        if k !='bald':
            for j,person in enumerate(data[k]):
                base_dir = '../full_data/new_updated/{}/{}/'.format(k,person)
                files = os.listdir(base_dir)
                files = [base_dir + file for file in files if file.endswith('.jpg')]
                x +=files
                y += [i]*len(files)
        else:
            base_dir = '../full_data/new_updated/bald/Bald_test/'
            files = os.listdir(base_dir)
            files = [base_dir + file for file in files if file.endswith('.jpg')]
            x += files
            y += [i] * len(files)
    
    return x,y
class Dataset(data.Dataset):
    def __init__(self,X,y,is_train = True, ls_eps = 0):
        self.X = X
        self.y = y
        self.ls_eps = ls_eps
        self.is_train = is_train
        self.transform = get_transform(is_train)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        x = self.X[idx]
        y = self.y[idx]
        x = cv2.imread(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(image = x)["image"]
        x = np.rollaxis(x,-1,0) # H,W,C -> C,H,W

        y = to_onehot(y,num_classes)
        y = label_smoothing(y,self.ls_eps)

        data = {}
        data['x'] = torch.from_numpy(x.astype('float32'))
        data['y'] = torch.from_numpy(y.astype('float32'))
        return data