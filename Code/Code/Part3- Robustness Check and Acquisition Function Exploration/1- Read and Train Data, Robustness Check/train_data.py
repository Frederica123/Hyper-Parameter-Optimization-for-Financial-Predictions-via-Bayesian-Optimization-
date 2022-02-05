from clean_data import ReadData
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




class TrainDataLearning:
    def __init__(self, x_train, x_val, y_train, y_val,batch_size=252, num_epochs=11):
        self.x_train=x_train
        self.x_val=x_val
        self.y_train=y_train
        self.y_val=y_val
        #self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.num_epochs=num_epochs



    def train_learning(self,learning_rate):
        # Here we need to pre-define an objective function that we want to minmize
        lstm_param_dict = read_json('security_selection/model/lstm_t2v_test_params.json')
        # this step is to access the default hyperparameter setting of the LSTM model 
        del lstm_param_dict['structure'][0]
        # delete the unnecessary 'structure' item from the dictionary. 
        #Otherwise, in my initial trials, this 'structure' item would render an output error. (should correct in Github)
        lstm_param_dict['learning_rate'] = learning_rate

        lstm_param_dict['batch_size'] = self.batch_size
        lstm_param_dict['num_epochs'] = self.num_epochs
        # although other hyperparameters are not decision variables, it is important to intentionally 
        #tweak other hyperparameters a little bit
        # to accelerate the searching processing of HyperOpt for optimal learning rate. 
        # Otherwise, in my second and third trials, it would be really time-consuming to 
        # generate a sub-optimal learning rate result
        # the details and assumptions of 'batch_size' and 'num_epochs' can be viewed in below Assumption part. 
        lstm_model = LSTM(**lstm_param_dict)
        # Call the LSTM function to set the LSTM model with newly tuned hyperparameter configuration
        lstm_model.fit(self.x_train.reshape(-1, 1, self.x_train.shape[1]), self.y_train)
        y_predict_lstm = lstm_model.predict(self.x_val.reshape(-1, 1, self.x_val.shape[1]))
        # use the updated LSTM model to fit and predict the test data 
        rmse_lstm = mean_squared_error(self.y_val, y_predict_lstm, squared=False)
        # Only choose the RMSE to measure the perdiciting behavior
        # calcuate the rmse(rooted mean square error) to measure the prediction accuracy of LSTM model
        # Assumption: the lower the rmse, the better the performance of LSTM prediction
        return 1/rmse_lstm
        # So here the rmse is assumed to be the key metric that needs to be minimized in my case

class TrainDataLearningBatch:
    def __init__(self, x_train, x_val, y_train, y_val,num_epochs=11):
        self.x_train=x_train
        self.x_val=x_val
        self.y_train=y_train
        self.y_val=y_val
        #self.learning_rate=learning_rate
        #self.batch_size=batch_size
        self.num_epochs=num_epochs



    def train_learning(self,learning_rate,batch_size):
        # Here we need to pre-define an objective function that we want to minmize
        lstm_param_dict = read_json('security_selection/model/lstm_t2v_test_params.json')
        # this step is to access the default hyperparameter setting of the LSTM model 
        del lstm_param_dict['structure'][0]
        # delete the unnecessary 'structure' item from the dictionary. 
        #Otherwise, in my initial trials, this 'structure' item would render an output error. (should correct in Github)
        lstm_param_dict['learning_rate'] = learning_rate

        lstm_param_dict['batch_size'] =int(batch_size)
        lstm_param_dict['num_epochs'] = self.num_epochs
        # although other hyperparameters are not decision variables, it is important to intentionally 
        #tweak other hyperparameters a little bit
        # to accelerate the searching processing of HyperOpt for optimal learning rate. 
        # Otherwise, in my second and third trials, it would be really time-consuming to 
        # generate a sub-optimal learning rate result
        # the details and assumptions of 'batch_size' and 'num_epochs' can be viewed in below Assumption part. 
        lstm_model = LSTM(**lstm_param_dict)
        # Call the LSTM function to set the LSTM model with newly tuned hyperparameter configuration
        lstm_model.fit(self.x_train.reshape(-1, 1, self.x_train.shape[1]), self.y_train)
        y_predict_lstm = lstm_model.predict(self.x_val.reshape(-1, 1, self.x_val.shape[1]))
        # use the updated LSTM model to fit and predict the test data 
        rmse_lstm = mean_squared_error(self.y_val, y_predict_lstm, squared=False)
        # Only choose the RMSE to measure the perdiciting behavior
        # calcuate the rmse(rooted mean square error) to measure the prediction accuracy of LSTM model
        # Assumption: the lower the rmse, the better the performance of LSTM prediction
        return 1/rmse_lstm
        # So here the rmse is assumed to be the key metric that needs to be minimized in my case
