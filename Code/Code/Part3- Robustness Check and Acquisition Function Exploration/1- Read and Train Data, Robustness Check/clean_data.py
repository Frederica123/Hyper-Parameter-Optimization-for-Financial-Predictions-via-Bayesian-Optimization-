#cd "C:\Users\lijing\OneDrive\4742\PersonalProject_OurBranch\WM-SecuritySelection"
### In my final report, all of import stuff that I used have been put in the same module in an alphabetical order
import csv


from datetime import datetime
from dateutil.relativedelta import relativedelta
from data.main import Pool, Meta
from data.reader.helpers import read_reader_params_from_json, read_performance_measures_params_from_json


import json

import matplotlib.pyplot as plt 
import numpy as np


import os
import pandas as pd 
import pickle  # used to save serialized files
import seaborn as sns
sns.set()
from security_selection.feature.main import Feature
from security_selection.feature.performance_measures import *

from security_selection.training.main import TrainingEnv

from security_selection.model.base import EstimatorBased, TransformerBased
from security_selection.model.sequence_model import LSTM, sequence_model_input_shape_map, sequence_model_single_dim_instance_prediction_input_shape_map

from security_selection.training.helpers import *

from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import security_selection.configuration as conf
import sys 
import tqdm

from tqdm import tqdm

import time
from utils import read_json, write_json

import warnings




class ReadData:
    def __init__(self,filename, pm_window=3,sample_window=30,test_start_date = '2020-06-30'):
        self.filepath=filename
        self.pm_window=pm_window
        self.sample_window=sample_window
        self.test_start_date=test_start_date
        self.lb_window=int(3 * self.pm_window * 365.25) + 1

            # performance measures window: number of years
    

   
    def prepare_data_for_er_ari(self):
        #filepath = "C:/Users/lijing/OneDrive/4742/PersonalProject_OurBranch/WM-SecuritySelection/security_selection/automation/notebooks/MF_LargeCap_ExcessReturn_3Y.parquet"
        er_ari_df = pd.read_parquet(self.filepath)
        

        data_dict = {ticker: er_ari_df[ticker].dropna() for ticker in er_ari_df.columns}
        tickers_to_remove = []
        
        label_dict = {}
        for ticker, series in tqdm(data_dict.items()):
            if series.isna().sum() == series.shape[0]:
                tickers_to_remove += [ticker]
                continue

            last_date = series.index[-1] - relativedelta(years=self.pm_window)
            if last_date <= series.index[0]:
                tickers_to_remove.append(ticker)
                continue

            index = series.loc[:series.index[-1] - relativedelta(years=self.pm_window)].index
            label_dict[ticker] = pd.Series([
                series[date + relativedelta(years=self.pm_window)] for date in index
            ], index=index)
            
        _ = [data_dict.pop(ticker) for ticker in tickers_to_remove]
        
        return data_dict, label_dict
    
    def train_test_data(self, data_dict,label_dict):
        tickers = list(data_dict.keys())
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []
        feat_window = 90

        
        # test start date
        checkpoint = datetime.strptime(self.test_start_date, '%Y-%m-%d') - relativedelta(years=self.pm_window)

        for ticker in tqdm(tickers):    
            label = label_dict[ticker]
            if label.shape[0] == 0:
                continue
            ts = data_dict[ticker].loc[:label.index[-1]]

            indices = [np.arange(i, i+self.lb_window, feat_window) for i in range(0, ts.shape[0] - self.lb_window + 1, self.sample_window)]
            
            temp_data = np.array([ts.iloc[sub_indices].values for sub_indices in indices])
            if temp_data.shape[0] == 0:
                continue
            temp_labels = np.array([label.loc[ts.index[sub_indices[-1]]] for sub_indices in indices])
            
            train_indices = [idx for idx in range(temp_data.shape[0]) if ts.index[indices[idx][-1]] <= checkpoint]
            test_indices = [idx for idx in range(temp_data.shape[0]) if ts.index[indices[idx][-1]] > checkpoint]
            
            train_data += [temp_data[train_indices]] 
            train_labels += [temp_labels[train_indices]]
            
            test_data += [temp_data[test_indices]] 
            test_labels += [temp_labels[test_indices]]
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)

        test_data = np.concatenate(test_data)
        test_labels = np.concatenate(test_labels)
        x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, train_size=0.9)

        return x_train, x_val, y_train, y_val