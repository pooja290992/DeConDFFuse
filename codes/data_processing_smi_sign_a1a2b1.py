import pandas as pd
import numpy as np 
import os 
from datetime import date
from sklearn.model_selection import train_test_split
import math
import torch.nn.functional as F
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import config as cfg
import torch

base_path = '../'
with open('../data/cols_list_smi_sign_a1a2b1.pkl', 'rb') as f: 
    columns_list = pickle.load(f)

def getData(file_name, sep = ','):
    data_df = pd.read_csv(base_path + 'data/' + file_name, sep = sep)
    return data_df


def getMolFeatsList():
    with open('data_mol_feats_columns.pickle', 'rb') as handle:
        feats_list = pickle.load(handle)
    return feats_list


def toFloatTensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


class DrugsData(Dataset):
    def __init__(self,train,labels):
        self.train_data = train
        self.labels = labels
        self.total_samples = len(self.train_data)
#        self.feats_list = feats_list
#         print('hey lable shape :', self.labels.shape)
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx):
#         print('idx : ',idx)
        db_id1 = self.train_data.iat[idx,0]
        db_id2 = self.train_data.iat[idx,1]
        db_data1 = cfg.data_df.loc[cfg.data_df['DrugBank ID'] == db_id1].reset_index(drop = True)
        db_data2 = cfg.data_df.loc[cfg.data_df['DrugBank ID'] == db_id2].reset_index(drop = True)

        db_data1.drop(['DrugBank ID'], axis = 1, inplace = True)
        db_data2.drop(['DrugBank ID'], axis = 1, inplace = True)
        db_data1 = db_data1[columns_list]
        db_data2 = db_data2[columns_list]
        db_data1 = db_data1.astype(float)
        db_data2 = db_data2.astype(float)
        return toFloatTensor(db_data1.values), toFloatTensor(db_data2.values), self.labels.iat[idx,0]
    

    
def getTrainTestDataFromFile():
    X_train = pd.read_csv('../data/X_train_new.csv')
    X_train = X_train.drop('Unnamed: 0',axis = 1)

    Y_train = pd.read_csv('../data/Y_train_new.csv')
    Y_train = Y_train.drop('Unnamed: 0',axis = 1)

    X_test = pd.read_csv('../data/X_test_new.csv')
    X_test = X_test.drop('Unnamed: 0',axis = 1)

    Y_test = pd.read_csv('../data/Y_test_new.csv')
    Y_test = Y_test.drop('Unnamed: 0',axis = 1)

    return X_train, X_test, Y_train, Y_test
    
    

    
