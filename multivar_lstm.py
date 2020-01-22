import pandas as pd
import numpy as np

import datetime
from dateutil.relativedelta import relativedelta

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import itertools

import os

import math


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers import Dropout

from keras.models import Model

from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback

from keras.models import model_from_json


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))





from basedeep import BaseDeepClass


class MultivariateLSTM(BaseDeepClass):
    
    
    def __init__(self, n_timesteps=168, n_preds = 20, standardizing_flag = False, scale_range = (-1,1), seasonal_lags_flag = False,
                 external_features_flag = False, use_features_per_lag_flag = False, n_batch_size = 512, n_epochs = 150, 
                 learning_rate = 0.001, adaptive_learning_rate_flag = False, early_stopping_flag = False, shuffle_flag=True,
                 dropout_rate = 0.3, regularizer_lambda1 = 0, regularizer_lambda2 = 0, clip_norm = True, 
                 n_hidden_neurons_1 = 256, n_hidden_neurons_2 = 32, stacked_flag = True):
     

        '''
        Variable explanation:
            >> n_timesteps = Number of lags used for sequences in LSTM network
            >> n_preds = Number of predictions made, "1" = prediction of next time step
            
            >> standardizing_flag: if == True -> input is standardized based on training set, 
                                   if False, scaling is done based on given scale_range
            >> scale_range: Range to scale input data
            
            >> seasonal_lags_flag: Indicates whether additional features (besides regular lags) should be created 
                                    [Note: Used to be called "additional_features_flag"], 
                                    currently: if set to "True", "seasonal lags" are be used or not
                                    
            >> external_features_flag: > if set to "True" incorporates available features into the model as additional feature vector 
                                       through a dense layer
                                        --> e.g. if seasonal_lags_flag was set to "True" and external_features_flag = True
                                            the seasonal_lags are incorporated in the model through an additional vector
                                       > if set to "False" indicates that a tensor is created to incorporate 
                                         additional features in the model --> in case additional features were created e.g. through 
                                         "use_features_per_lag_flag" and/or "seasonal_lags_flag"
                                       
            >> use_features_per_lag_flag: whether to create additional features for the lags itself (e.g. hour encoded feature for the lag 1, 2, 3,...)
        
            >> n_batch_size: batch_size for training the model
            >> n_epochs: epochs for training the model   
            >> learning_rate: learning_rate for training the model 
            >> adaptive_learning_rate_flag: Flag indicates whether learning_rate should be decreased proportional to epochs
        
            >> n_hidden_neurons_1: number of neurons in First LSTM Layer (if only single Layer LSTM is used, 
                                    n_hidden_neurons_1, indicate number of neurons used in this Network)
            >> n_hidden_neurons_2: number of neurons in Second LSTM Layer 
            
            >> stacked_flag: indicates whether a model with 2 hidden layers (stacked_flag == "True") 
                             or a model with one hidden layer (stacked_flag == False) is selected
            
            >> regularizer_lambda1: Regularizer for Lasso
            >> regularizer_lambda2: Regularizer for Ridge
            
            >> dropout_rate: dropout_rate for Regularization  
            >> clip_norm: Flag to indicate whether Clipping of Gradients should be applied
            
            >> early_stopping_flag: Flag to indicate whether Early_stopping of training should be applied: 
                                    Note: currently set to 20 epochs 
                                    --> Trainings stops, if evaluationmetric does not improve for 20 epochs 
                                    
            >> shuffle_flag: Indicates whether shuffling of input data during training should be applied
            
            
            >> prediction_model: prediction_model trained with input data and given parameters
            >> actuals: actuals values of time series used to split data into training/valid/test sets, 
                        invert differencing & calculate RMSE between actuals and predictions
                        Note: for MultLSTM actuals = DataFrame
                        
            >> model_name: name of model given during training
            
            >> training_history: most recent history of training    
                        
        '''
        

        self.n_timesteps = n_timesteps
        self.n_preds = n_preds
        
        self.standardizing_flag = standardizing_flag
        self.scale_range = scale_range
        
        self.seasonal_lags_flag = seasonal_lags_flag
        self.external_features_flag = external_features_flag
        self.use_features_per_lag_flag = use_features_per_lag_flag
        
        self.n_batch_size = n_batch_size 
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate 
        self.adaptive_learning_rate_flag = adaptive_learning_rate_flag
                        
        self.n_hidden_neurons_1 = n_hidden_neurons_1
        self.n_hidden_neurons_2 = n_hidden_neurons_2
        self.stacked_flag = stacked_flag

        self.regularizer_lambda1 = regularizer_lambda1
        self.regularizer_lambda2 = regularizer_lambda2
        self.dropout_rate = dropout_rate
        self.clip_norm = clip_norm
        
        self.early_stopping_flag = early_stopping_flag
        self.shuffle_flag = shuffle_flag

        self.prediction_model = None
        self.actuals = None
        
        self.model_name = None
        self.training_history = None

        
        
        
    def get_params(self):
        '''
        Returns all parameters of model
        '''
        
        param_dict = {"n_timesteps": self.n_timesteps, 
                      "n_preds": self.n_preds, 
                      "standardizing_flag": self.standardizing_flag,
                      "scale_range" : self.scale_range,
                      "seasonal_lags_flag" : self.seasonal_lags_flag,
                      "external_features_flag" : self.external_features_flag,
                      "n_batch_size" : self.n_batch_size,
                      "n_epochs" : self.n_epochs,
                      "learning_rate" : self.learning_rate,
                      "adaptive_learning_rate_flag" : self.adaptive_learning_rate_flag,
                      "n_hidden_neurons_1" : self.n_hidden_neurons_1,
                      "n_hidden_neurons_2" : self.n_hidden_neurons_2,
                      "stacked_flag" : self.stacked_flag,
                      "regularizer_lambda1" : self.regularizer_lambda1,
                      "regularizer_lambda2" : self.regularizer_lambda2,
                      "dropout_rate" : self.dropout_rate,
                      "clip_norm" : self.clip_norm,
                      "early_stopping_flag" : self.early_stopping_flag,
                      "shuffle_flag" : self.shuffle_flag,
                      "seasonal_lags_flag" : self.seasonal_lags_flag,
                      "prediction_model" : self.prediction_model,
                      "actuals" : self.actuals
                     }
                     
        return param_dict
    
  



   
    def load_model(self, model_to_load):
        
        '''
        function "loads" Keras model which was stored on disk into Class
        
        Note: this only works if params of Class are the same as params used to train "model_to_load"
        
        '''
        
        print('Note: "loaded" model must have identical params as the Class currently has')
        
        self.prediction_model = model_to_load

    
          
        
              
        
    def get_features_and_supervised_data_dict(self, ts_series, seasonal_lag_set, start_train_year, 
                                              last_train_set_year, st_valid_year, st_test_year, 
                                              end_valid_year=None, end_test_year=None, verbose=0):               
        
        '''
        >> function creates dfs for EACH area with taxi_requests and additional features like: encoded weekofday, lags...
        >> dfs are stored in dict
        >> function returns two dicts:  1) dict with actuals, lags, seasonal lags of target & addtional features of target  
                                        2) dict with features of the lags (different encodings like weekofday.. for each lag value)

        '''
                
        
        #assign years correctly:
        if end_valid_year == None:
            end_valid_year = st_valid_year
        
        if st_test_year == None:
            st_test_year = st_valid_year
            
        if end_test_year == None:
            end_test_year = st_test_year
            
            
        if verbose == 1:
            print('generate data..')
            print('st_valid_year: ', st_valid_year)
            print('end_valid_year: ', end_valid_year)
            print('st_test_year: ', st_test_year)
            print('end_test_year: ', end_test_year)
        
        
        ts_series_copy = ts_series.copy()
        #create DataFrame: (necessary since ts_series might be a pd.Series which does not have 'columns' parameter )
        ts_series_copy = pd.DataFrame(ts_series_copy)

        #store column labels:
        area_labels = list(ts_series_copy.columns)

        #store results in dict:
        areas_encoded_dict = {}
        areas_window_lags_feats_dict = {}

        # append weekofday encoding, hourofday encoding, monthofyear encoding, lags features:
        #Note: lags have to be appended on last step, since we drop rows with no valid lags (NaNs)

        #for each area:
        for i in range(len(area_labels)):
            
            if verbose ==1:
                print('create data of area ', area_labels[i])

            # 1) slice actuals of ts_series for each area:
            areas_encoded_dict['area{}'.format(area_labels[i])] = ts_series_copy.iloc[:,i]

            # 2) append weekofday encoded features
            weekofday_dummies_area_i = self.get_day_of_week_features_one_hot_encoded(ts_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             weekofday_dummies_area_i],axis=1)

            # 3) append hourofday encoding
            hourofdays_encod_area_i = self.get_hour_of_day_features_sin_cos_encoded(ts_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             hourofdays_encod_area_i],axis=1)

            # 4) append monthofyear encoding
            monthofyear_encod_area_i = self.get_month_of_year_features_sin_cos_encoded(ts_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             monthofyear_encod_area_i],axis=1)

            
            
            if self.use_features_per_lag_flag == True:
                
                print('create additional tensor for lags...')
                #get encoding for each observation (lag) --> create sequence of encodings
                #e.g. sin-month encoding for observation1, observation2,...observation168, (lagged observations)
                #create a sequence such that sin_month encoding1, sin_month encoding2, ... sin_month encoding168... is obtained
                # --> creating this sequence for all lags, a tensor can be obtained which can be processed by the LSTM
                #tensor shape: number of observations x n_timestamps (number of lagged values) x features
                
                #create matrix of lags for each feature: (create a sequence such that sin_month encoding1, sin_month encoding2, ... sin_month encoding168... is obtained)
                prev_lag_encodings_dict_area_i = self.get_encoded_feature_matrix_for_prev_lags(areas_encoded_dict['area{}'.format(area_labels[i])].iloc[:,1:], 
                                                                                          self.n_timesteps, verbose = verbose)
                
                #based on feature matrix create 3D tensor with correct shape of sequences & based on correct dates of training/valid/ test set
                tensor_list_area_i = self.get_encoding_tensor(prev_lag_encodings_dict_area_i, start_train_year, last_train_set_year,
                                                         start_valid_year=st_valid_year, end_valid_year=end_valid_year,
                                                         start_test_year=st_test_year, end_test_year=end_test_year, verbose=verbose)
        

                #create dict to store results:
                areas_window_lags_feats_dict['area{}'.format(area_labels[i])] = []
                #get encodings for each data set (year):
                for u in range(len(tensor_list_area_i)):
                    #append array to new dict:
                    areas_window_lags_feats_dict['area{}'.format(area_labels[i])].append(tensor_list_area_i[u])
                    

            else:
                #return empty dict:
                areas_window_lags_feats_dict = {}

        
        #print('## areas_encoded_dict')
            
        #print(areas_encoded_dict['area237'])
        
        #print('## areas_window_lag')
        #print(areas_window_lags_feats_dict['area237'][0])
        #print(areas_window_lags_feats_dict['area237'][0][0])

        
        # 6) append lagged (seasonal lags & sliding window) and get final dataset for each area with all features:
        #-> this way we already get a "supervised" dataset very efficiently
        for i in range(len(area_labels)):
            #call function to create supervised dataset with lags:
            ts_all_featrs = self.create_supervised_data_single_ts(areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                  self.n_timesteps, self.seasonal_lags_flag, seasonal_lag_set)

            #assign final dataset of each area to dict:
            areas_encoded_dict['area{}'.format(area_labels[i])] = ts_all_featrs
            
            
        #7) create tensor of seasonal lag features:
        if self.seasonal_lags_flag == True:
            for i in range(len(area_labels)):
                
                #get feature matrix of seasonal lags: 
                prev_lag_seasonal_encodings_dict_area_i = self.get_encoded_feature_matrix_for_prev_lags(areas_encoded_dict['area{}'.format(area_labels[i])].iloc[:,-len(seasonal_lag_set):], 
                                                                                          self.n_timesteps, verbose = verbose)
                                                
                #create tensor for seasonal feautres --> this way a sequence for each seasonal lag value is created (within the tensor)
                # --> the tensor can be concatenated on axis=2 to obtain a 3D tensor which can be used as input for the LSTM model 
                # --> without a sequence of features, the LSTM model can not process the additional features
                
                #based on feature matrix create 3D tensor with correct shape of sequences & based on correct dates of training/valid/ test set
                tensor_list_seasonal_feat_area_i = self.get_encoding_tensor(prev_lag_seasonal_encodings_dict_area_i, start_train_year, last_train_set_year,
                                                         start_valid_year=st_valid_year, end_valid_year=end_valid_year,
                                                         start_test_year=st_test_year, end_test_year=end_test_year, verbose=verbose)
                
                if verbose > 1:
                    print('shape of seasonal tensor train set: ', tensor_list_seasonal_feat_area_i[0].shape)
                
                #concat features with other lagged feature tensor if it exists:
                if areas_window_lags_feats_dict:
                    for u in range(len(tensor_list_seasonal_feat_area_i)):
                        existing_tensor = areas_window_lags_feats_dict['area{}'.format(area_labels[i])][u]
                        existing_tensor = np.concatenate((existing_tensor, tensor_list_seasonal_feat_area_i[u]), axis=2)
                        #store tensor again:
                        areas_window_lags_feats_dict['area{}'.format(area_labels[i])][u] = existing_tensor
                        
                else:
                    areas_window_lags_feats_dict['area{}'.format(area_labels[i])] = []
                    #get encodings for each data set (year):
                    for u in range(len(tensor_list_seasonal_feat_area_i)):
                        #append array to new dict:
                        areas_window_lags_feats_dict['area{}'.format(area_labels[i])].append(tensor_list_seasonal_feat_area_i[u])

        #return "areas_encoded_dict" with actuals, lags, and encodings + seasonal lags of target 
        #  & return "areas_window_lags_feats_dict" with encodings of lags of each window
        return areas_encoded_dict, areas_window_lags_feats_dict
    
    
    
        
        
    def generate_data_and_features_multivar_ts(self, multivar_series, start_train_year, last_train_set_year, 
                                               start_validation_set_year, start_test_set_year, 
                                               end_validation_set_year=None, end_test_set_year=None, verbose=0):               
        
        '''
        
        #prepares data to be used as input for models (scaling of each area, creation of features 
        by calling some other functions, concatenation, reshaping..) 
        
        '''       
        
        # 1) apply differencing
        ts_diff = self.differencing(multivar_series,1)
        #print(ts_diff)

        # 2) get df for each area with encoded features appended:
        if self.seasonal_lags_flag == True:
            lag_set = [168,336,504,672]
        else:
            lag_set = [] #if seasonal lags are not used, return empty list

        #get supervised data & additional features per area:
        #call function to generate features for each area:                                               
        areas_encoded_dict, areas_window_lags_feats_dict = self.get_features_and_supervised_data_dict(ts_diff, lag_set,
                                                                                start_train_year,
                                                                                last_train_set_year,
                                                                                start_validation_set_year, 
                                                                                start_test_set_year, 
                                                                                end_valid_year = end_validation_set_year,
                                                                                end_test_year = end_test_set_year,
                                                                                verbose=verbose)          

        # 3) get Train/Test-Split for each area & scale data:
        #create dict to store results:
        supervised_data_dict = {}
        #create list to store scaler of each area:
        scaler_list = []
        #for each area create train/test split & scale data & append to dict:
        for key in areas_encoded_dict:
            #get train/validation/test split:    
            
            #assign years correctly:
            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year
            
            if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
                
            if end_test_set_year == None:
                end_test_set_year = start_test_set_year
            
            ts_train = areas_encoded_dict[key].loc[start_train_year:last_train_set_year]
            ts_test = areas_encoded_dict[key].loc[start_test_set_year:end_test_set_year]             
            ts_valid = areas_encoded_dict[key].loc[start_validation_set_year:end_validation_set_year]

            '''
            Note: create final valid_set by adding last window of ts_train is not needed since 
                    we created supervised data with shift() function'''

            #slice first column of each df & lagged features (stored in last columns) -> need to be scaled:
            ts_train_slice = pd.concat([pd.DataFrame(ts_train.iloc[:,0]),ts_train.iloc[:,-(self.n_timesteps+len(lag_set)):]],axis=1)
            ts_valid_slice = pd.concat([pd.DataFrame(ts_valid.iloc[:,0]),ts_valid.iloc[:,-(self.n_timesteps+len(lag_set)):]],axis=1)         
            ts_test_slice = pd.concat([pd.DataFrame(ts_test.iloc[:,0]),ts_test.iloc[:,-(self.n_timesteps+len(lag_set)):]],axis=1)

            #create arrays for scaler:
            ts_train_slice = ts_train_slice.values
            ts_valid_slice = ts_valid_slice.values
            ts_test_slice = ts_test_slice.values

            # Apply scaling:
            #Note: data_scaler() returns dfs!!  
            if verbose == 1:
                print('Data is scaled...')
            #scale data:
            scaler, train_scaled, valid_scaled, test_scaled = self.data_scaler(ts_train_slice, ts_valid_slice, ts_test_slice,
                                                                               self.scale_range, self.standardizing_flag, verbose)

            #restore index: --> necessary to concat dfs correctly
            train_scaled.index = ts_train.index
            valid_scaled.index = ts_valid.index
            test_scaled.index = ts_test.index

            #append X,y pairs to dict:  
            if self.seasonal_lags_flag == True:
                #slice lags (n_timesteps) to obtain X_data
                X_train = train_scaled.iloc[:,1:(-len(lag_set))].values
                X_valid  = valid_scaled.iloc[:,1:(-len(lag_set))].values                   
                X_test = test_scaled.iloc[:,1:(-len(lag_set))].values
                #first column of scaled data = actuals       
                y_train = train_scaled.iloc[:,0].values       
                y_valid = valid_scaled.iloc[:,0].values
                y_test = test_scaled.iloc[:,0].values 
                
                #slice additional generated features which can be fused to the model through an extra vector:

                #slice time features of target (hour of week encoded, sin cos encoding.. of target) : 
                train_target_time_features = ts_train.iloc[:,1:-(self.n_timesteps+len(lag_set))].values
                valid_target_time_features = ts_valid.iloc[:,1:-(self.n_timesteps+len(lag_set))].values
                test_target_time_features = ts_test.iloc[:,1:-(self.n_timesteps+len(lag_set))].values
                #slice seasonal_lags of target:
                train_season_lags = ts_train.iloc[:,-len(lag_set):].values
                valid_season_lags = ts_valid.iloc[:,-len(lag_set):].values
                test_season_lags = ts_test.iloc[:,-len(lag_set):].values 

            else:
                #slice lags (n_timesteps) to obtain X_data
                X_train = train_scaled.iloc[:,1:].values
                X_valid  = valid_scaled.iloc[:,1:].values                   
                X_test = test_scaled.iloc[:,1:].values
                #first column of scaled data = actuals       
                y_train = train_scaled.iloc[:,0].values       
                y_valid = valid_scaled.iloc[:,0].values
                y_test = test_scaled.iloc[:,0].values 
                
                #slice additional generated features which can be fused to the model through an extra vector:
                
                #slice time features of target (hour of week encoded, sin cos encoding.. of target) : 
                train_target_time_features = ts_train.iloc[:,1:-(self.n_timesteps)].values
                valid_target_time_features = ts_valid.iloc[:,1:-(self.n_timesteps)].values
                test_target_time_features = ts_test.iloc[:,1:-(self.n_timesteps)].values
                #slice seasonal_lags of target:
                train_season_lags = []
                valid_season_lags = []
                test_season_lags = [] 


            #reshape X and y data to easily append other areas later on:
            list_to_reshape_X = [X_train, X_valid, X_test] 
            list_to_reshape_y = [y_train, y_valid, y_test]
            reshaped_list_X = []
            reshaped_list_y = []
            #reshape X:
            for array in list_to_reshape_X:
                #shape: (#n_samples,n_lags,n_areas)
                array = array.reshape((array.shape[0],array.shape[1],1))
                reshaped_list_X.append(array)

            #reshape y:
            for array in list_to_reshape_y:
                #shape: (#n_samples,n_areas)
                array = array.reshape((array.shape[0],1))
                reshaped_list_y.append(array)

            #append X,y to dict:
            supervised_data_dict[key] = []
            supervised_data_dict[key].append(reshaped_list_X[0])
            supervised_data_dict[key].append(reshaped_list_y[0])
            supervised_data_dict[key].append(reshaped_list_X[1])
            supervised_data_dict[key].append(reshaped_list_y[1])
            supervised_data_dict[key].append(reshaped_list_X[2])
            supervised_data_dict[key].append(reshaped_list_y[2])
            #append seasonal lags:
            '''NOTE: !!! -> so far seasonal lags are not used for multivar model.. 
                            not intuitive how to concatenate the features since lags differ between areas'''
            supervised_data_dict[key].append(train_season_lags)
            supervised_data_dict[key].append(valid_season_lags)
            supervised_data_dict[key].append(test_season_lags)
            #append time features of target:
            supervised_data_dict[key].append(train_target_time_features)
            supervised_data_dict[key].append(valid_target_time_features)
            supervised_data_dict[key].append(test_target_time_features)


            #append scaler of each area to list:
            scaler_list.append(scaler)

        #quick check if shape is correct:
        if verbose == 1:
            print('X_train shape of area237 before concat with other areas: ', supervised_data_dict['area237'][0].shape)
            print('X_valid shape of area237 before concat with other areas: ', supervised_data_dict['area237'][2].shape)
            print('X_test shape of area237 before concat with other areas: ', supervised_data_dict['area237'][4].shape)
            print('y_train shape of area237 before concat with other areas: ', supervised_data_dict['area237'][1].shape)
            print('y_valid shape of area237 before concat with other areas: ', supervised_data_dict['area237'][3].shape)
            print('y_test shape of area237 before concat with other areas: ', supervised_data_dict['area237'][5].shape)


        #create training set, valid & test set containing inputs of all selected areas: -> append all areas into one big np.array!
        key_list = list(supervised_data_dict.keys())

        #fill arrays with entries of first area:
        X_train, y_train = supervised_data_dict[key_list[0]][0], supervised_data_dict[key_list[0]][1]
        X_valid, y_valid = supervised_data_dict[key_list[0]][2], supervised_data_dict[key_list[0]][3]
        X_test, y_test = supervised_data_dict[key_list[0]][4], supervised_data_dict[key_list[0]][5]
        #concat remaining areas:
        for i in range(1,len(key_list)):
            X_train = np.concatenate((X_train,supervised_data_dict[key_list[i]][0]),axis=2)
            X_valid = np.concatenate((X_valid,supervised_data_dict[key_list[i]][2]),axis=2)
            X_test = np.concatenate((X_test,supervised_data_dict[key_list[i]][4]),axis=2)
            y_train = np.concatenate((y_train,supervised_data_dict[key_list[i]][1]),axis=1)
            y_valid = np.concatenate((y_valid,supervised_data_dict[key_list[i]][3]),axis=1)
            y_test = np.concatenate((y_test,supervised_data_dict[key_list[i]][5]),axis=1)
            
            #concat tensor of seasonal lags if tensor is available:
            if self.seasonal_lags_flag == True and self.use_features_per_lag_flag == False and self.external_features_flag == False:
                X_train = np.concatenate((X_train,areas_window_lags_feats_dict[key_list[i]][0]),axis=2)
                X_valid = np.concatenate((X_valid,areas_window_lags_feats_dict[key_list[i]][1]),axis=2)
                X_test = np.concatenate((X_test,areas_window_lags_feats_dict[key_list[i]][2]),axis=2)

        #create features which can be used as external feature vector:
        '''Note: since time encoded features of target (like "sin cos of month/hour & dayofweek") are equal among all time series, 
                it is sufficient to store the encoded features of one area --> here area"0"  
           Note: "train_season_lags", "valid_season_lags",..., per area are currently not returned by function 
                  --> Thus, not available to be used as an additional feature vector
                
        '''
        target_time_encoded_featrs_train = supervised_data_dict[key_list[0]][9]
        target_time_encoded_featrs_valid = supervised_data_dict[key_list[0]][10]
        target_time_encoded_featrs_test = supervised_data_dict[key_list[0]][11]


        '''#Note: if flag is set to true, features of each timestep within each time window is concatenated on axis=2 
                    (similar to Xu et al 2018: for each lag we add addtional features on axis=2)     '''                          
        if self.use_features_per_lag_flag == True:
            if verbose == 1:
                print('Addtional features of each lag are concatenated on axis = 2 ')
                #append features of each area: 
                '''#Note: since addtional features of each lag are the same among all areas, 
                            it is sufficient to only use features of one area: '''
                print('shape of tensor of lags for train set:', areas_window_lags_feats_dict[key_list[0]][0].shape)
                print('Xtrain_shape ' , X_train.shape)
                
            #concatenate X_train with corresponding window lags for each set (train,valid,test) stored in 
            #     areas_window_lags_feats_dict          
            X_train = np.concatenate((X_train,areas_window_lags_feats_dict[key_list[0]][0]),axis=2)
            X_valid = np.concatenate((X_valid,areas_window_lags_feats_dict[key_list[0]][1]),axis=2)
            X_test = np.concatenate((X_test,areas_window_lags_feats_dict[key_list[0]][2]),axis=2)

        #print final shapes:
        if verbose == 1:
            list_to_print_shapes = [X_train,X_valid,X_test,y_train,y_valid,y_test]
            list_to_print_labels = ['X_train','X_valid','X_test','y_train','y_valid','y_test']
            for i in range(len(list_to_print_shapes)):
                print('final concatenated shape of {}: '.format(list_to_print_labels[i]), list_to_print_shapes[i].shape)

        #double check scaler type:
        if verbose == 1:
            print('scaler type: ', scaler_list[0])

        #Note: necessary to return scaler: we need scaler to invert scaling for predictions. 
        return (X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, target_time_encoded_featrs_train,
                target_time_encoded_featrs_valid, target_time_encoded_featrs_test)           


                                         

        
        
    def generate_data(self, multivar_series, start_train_year, last_train_set_year, start_validation_set_year, 
                      start_test_set_year, end_validation_set_year=None, end_test_set_year=None,verbose=0):

        '''
        "super-function" which calls above functions (generate_data_and_features_multivar_ts() & 
        get_features_and_supervised_data_dict() ) to generate features and get train/test split & 
        returns everything in right shape and format
        
        '''
          
        '''# for each area: generate supervised data (target with its lags arranged in df to diretly be used in Model) 
            & get train/test split & scale data --> Then concatenate all areas to return X,y pairs for all areas:'''

        #call function to generate data for each area & concatenate results to receive final data sets for modeling:
        X_tr, y_tr, X_val, y_val, X_te, y_te, s_list, t_train, t_valid, t_test = self.generate_data_and_features_multivar_ts(
                                                                                        multivar_series,
                                                                                        start_train_year,
                                                                                        last_train_set_year,
                                                                                        start_validation_set_year, 
                                                                                        start_test_set_year, 
                                                                                     end_validation_set_year=end_validation_set_year,
                                                                                        end_test_set_year=end_test_set_year,
                                                                                        verbose=verbose)               

        
        #assign variable_names correctly: (if variable names are used above the line would be too long!)
        X_train = X_tr
        y_train = y_tr
        X_valid = X_val
        y_valid = y_val
        X_test = X_te
        y_test = y_te
        scaler_list = s_list
        target_time_encoded_featrs_train = t_train
        target_time_encoded_featrs_valid = t_valid
        target_time_encoded_featrs_test = t_test
        
        
        #check if addtional features of target should be used or not:
        if self.external_features_flag == True:
            #create list to store target_time_encoded_featrs:
            target_featrs_list = [target_time_encoded_featrs_train,target_time_encoded_featrs_valid, target_time_encoded_featrs_test]

        else:
            #create empty list to return, since we don't use features of target:
            target_featrs_list = [] 


        #Note: necessary to return scaler: we need scaler to invert scaling for predictions. 
        return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, target_featrs_list


                             
        
        
    def generate_data_get_predictions(self, multivar_series_actuals, start_train_year, last_train_set_year, 
                                      start_validation_set_year, start_test_set_year, 
                                      end_validation_set_year=None, end_test_set_year=None, verbose=0):
        
        '''
         function that creates data for multivar model that are only used for prediction task NOT training:
        
        '''
        
        '''#call function to create data: --> actually only X_valid and X_test are used for predictions 
            but training data is needed since valid & test sets have to be scaled acoording to training set'''
        X_tr, y_tr, X_val, y_val, X_te, y_te, s_list, target_featrs_list = self.generate_data(multivar_series_actuals,
                                                                            start_train_year, 
                                                                            last_train_set_year, 
                                                                            start_validation_set_year, 
                                                                            start_test_set_year,
                                                                            end_validation_set_year=end_validation_set_year,
                                                                            end_test_set_year=end_test_set_year,    
                                                                            verbose=verbose)
        
        
        #assign variable_names correctly: (if variable names are used above the line would be too long!)
        X_train = X_tr
        y_train = y_tr
        X_valid = X_val
        y_valid = y_val
        X_test = X_te
        y_test = y_te
        scaler_list = s_list

        
        multivariate_flag = True
        
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        #prediction results validation data:
        validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.seasonal_lags_flag, target_featrs_list,
                                                                          multivariate_flag, self.n_preds, start_validation_set_year,
                                                                          end_validation_set_year, True, 
                                                                          scaler_list, self.standardizing_flag, self.scale_range,
                                                                          self.prediction_model, multivar_series_actuals,
                                                                          'results_{}'.format(start_validation_set_year), verbose)

        #prediction results test data:
        if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
                
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
            
        valid_flag = False
        prediction_results, rmse_results_test = self.get_preds_non_state(X_test, self.seasonal_lags_flag, target_featrs_list,
                                                                         multivariate_flag, self.n_preds, start_test_set_year, 
                                                                         end_test_set_year, valid_flag,
                                                                         scaler_list, self.standardizing_flag, self.scale_range, 
                                                                         self.prediction_model, multivar_series_actuals, 
                                                                         'results_{}'.format(start_test_set_year), verbose)                 


        return validation_results, prediction_results, rmse_results_valid, rmse_results_test



    
    
    # compiles and fits static (non-stateful) 1H LSTM Network
    def create_model_1H(self, X_train, y_train, X_valid, y_valid, external_features_list, model_name):   
        
            
        if self.regularizer_lambda1 != 0 or self.regularizer_lambda2 != 0:
            self.dropout_rate = 0 #set dropout_rate to zero if other regularization is used!

        if self.regularizer_lambda1 != 0 and self.regularizer_lambda2 == 0:
            print('#L1-regularization applied')
            regul_LX =regularizers.l1(self.regularizer_lambda1)

        elif self.regularizer_lambda2 != 0 and self.regularizer_lambda1 == 0:
            print('#L2-regularization applied')
            regul_LX =regularizers.l2(self.regularizer_lambda2)

        else:
            print('#Dropout applied')
            regul_LX =regularizers.l2(0) #set regularizer to zero since dropout is applied


        #define optimizer for model:
        if self.clip_norm == True:
            print('#Clipping Norm applied')
            adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0, clipnorm=1.0)
        else:
            adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0)


        #create callback list for different callbacks:
        callback_list = []

        #create custom callback to monitor learningrate:
        #Note: only monitors changing learning_rate if learning_rate is changed by LearningRateScheduler
        class LearningRateTracker(Callback):
            def on_epoch_end(self,epoch,logs={}):
                lr = K.eval(self.model.optimizer.lr)
                print('#Current LearningRate: ', lr)

        lr_tracker = LearningRateTracker()
        #append callback to list:
        callback_list.append(lr_tracker)   


        #set path to store best model:
        Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/Stacked_LSTM/'
                     'Hyperparam_tuning_y2011/Best_Models/')
        #create Callback to store best model during training (until training is stopped if EarlyStopping is applied) 
        #(-> model at earlystopping, and best model before earlystopping is applied might differ):
        model_check = ModelCheckpoint(Save_PATH + model_name +'_bestmodel.{epoch:02d}-{val_loss:.2f}.h5', 
                                      monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        #append callback to list:
        callback_list.append(model_check)


        if self.early_stopping_flag == True:
            print('#Early Stopping applied')
            #create Callback for EarlyStopping:
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            #append callback to list:
            callback_list.append(early_stop)

        if self.adaptive_learning_rate_flag == True:
            print('#Adaptive Learningrate applied')
            #create function for adaptive learning rate:
            def lr_step_decay(epoch, learning_rate):
                initial_lrate = learning_rate
                if epoch > 0:
                    lrate = initial_lrate/math.sqrt(epoch)
                    return lrate
                else: 
                    return initial_lrate

            #create Callback for Scheduler:
            lrate_schedule = LearningRateScheduler(lr_step_decay)
            callback_list.append(lrate_schedule)



        #definition of model: different models created depending on external input

        if self.external_features_flag == True:
            print('#1H-LSTM Model with External Features is created...')
            #Note: external_features_list contains at first index: all training data, 
            #      second index: all validation data, third index: all test data  

            #create 2D tensor of external features: 
            external_input = Input(shape=(external_features_list[0].shape[1],)) #take shape of number of features 

            #define model input tensor of sequences
            model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))

            #create layers of LSTM:
            lstm_layer1 = LSTM(self.n_hidden_neurons_1, activation='tanh', dropout=self.dropout_rate, 
                               kernel_regularizer=regul_LX)(model_input)

            #concatenate external featrs on axis=1 with output of last LSTM layer (output of LSTM layer = sequence representation)
            merged_featrs = Concatenate(axis=1)([lstm_layer1, external_input]) 
            #Dense layer takes merged values as input:
            model_dense = Dense(self.n_preds,activation='linear')(merged_featrs)

            #compile:
            lstm_1H_model = Model(inputs=[model_input, external_input], outputs=model_dense)
            lstm_1H_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])

            # fit network
            history = lstm_1H_model.fit([X_train, external_features_list[0]], y_train, epochs=self.n_epochs, 
                                        batch_size = self.n_batch_size, validation_data=([X_valid, external_features_list[1]],
                                                                                         y_valid), verbose=1, 
                                        shuffle = self.shuffle_flag, callbacks = callback_list) 


        if self.external_features_flag == False:
            print('Regular 1H-LSTM Model is created...')  

            # define model
            model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
            lstm_layer1 = LSTM(self.n_hidden_neurons_1, activation='tanh', dropout=self.dropout_rate, 
                               kernel_regularizer=regul_LX)(model_input)
            model_dense = Dense(self.n_preds,activation='linear')(lstm_layer1)

            #compile:
            lstm_1H_model = Model(inputs=[model_input], outputs=model_dense)
            lstm_1H_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])

            # fit network
            history = lstm_1H_model.fit(X_train, y_train, epochs=self.n_epochs, batch_size = self.n_batch_size, 
                                        validation_data=(X_valid, y_valid), verbose=1, shuffle = self.shuffle_flag, 
                                        callbacks=callback_list) 

        #assign/"store" model, history & model_name:
        self.prediction_model = lstm_1H_model 
        self.model_name = model_name
        self.training_history = history
        
        return history, lstm_1H_model

   
    
    

    #compiles and fits Stacked non-stateful LSTM
    def create_model_2H(self, X_train, y_train, X_valid, y_valid, external_features_list, model_name):
        
        if self.regularizer_lambda1 != 0 or self.regularizer_lambda2 != 0:
            self.dropout_rate = 0 #set dropout_rate to zero if other regularization is used!

        if self.regularizer_lambda1 != 0 and self.regularizer_lambda2 == 0:
            print('#L1-regularization applied')
            regul_LX =regularizers.l1(self.regularizer_lambda1)

        elif self.regularizer_lambda2 != 0 and self.regularizer_lambda1 == 0:
            print('#L2-regularization applied')
            regul_LX =regularizers.l2(self.regularizer_lambda2)

        else:
            print('#Dropout applied')
            regul_LX =regularizers.l2(0) #set regularizer to zero since dropout is applied

        #define optimizer for model:
        if self.clip_norm == True:
            print('#Clipping Norm applied')
            adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0, clipnorm=1.0)
        else:
            adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0)


        #create callback list for different callbacks:
        callback_list = []

        #create custom callback to monitor learningrate:
        #Note: only monitors changing learning_rate if learning_rate is changed by LearningRateScheduler
        class LearningRateTracker(Callback):
            def on_epoch_end(self,epoch,logs={}):
                lr = K.eval(self.model.optimizer.lr)
                print('#Current LearningRate: ', lr)

        lr_tracker = LearningRateTracker()
        #append callback to list:
        callback_list.append(lr_tracker)    


        #set path to store best model:
        Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/Stacked_LSTM/'
                     'Hyperparam_tuning_y2011/Best_Models/')
        #create Callback to store best model during training (until training is stopped if EarlyStopping is applied) 
        #      (-> model at earlystopping, and best model before earlystopping is applied might differ):
        model_check = ModelCheckpoint(Save_PATH + model_name +'_bestmodel.h5', monitor='val_loss', 
                                      mode='min', verbose=1, save_best_only=True)
        #append callback to list:
        callback_list.append(model_check)


        if self.early_stopping_flag == True:
            print('#Early Stopping applied')
            #create Callback for EarlyStopping:
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            #append callback to list:
            callback_list.append(early_stop)

        if self.adaptive_learning_rate_flag == True:
            print('#Adaptive Learningrate applied')
            #create function for adaptive learning rate:
            def lr_step_decay(epoch, learning_rate):
                initial_lrate = learning_rate
                if epoch > 0:
                    lrate = initial_lrate/math.sqrt(epoch)
                    return lrate
                else: 
                    return initial_lrate

            #create Callback for Scheduler:
            lrate_schedule = LearningRateScheduler(lr_step_decay)
            callback_list.append(lrate_schedule)      


        #definition of model: different models created depending on external input

        if self.external_features_flag == True:
            print('#2H-LSTM Model with External Features is created...')
            #Note: external_features_list contains at first index: all training data, 
            #      second index: all validation data, third index: all test data  

            #create 2D tensor of external features: 
            external_input = Input(shape=(external_features_list[0].shape[1],)) #take shape of number of features 

            #define model input tensor of sequences
            model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))

            #create layers of LSTM:
            lstm_layer1 = LSTM(self.n_hidden_neurons_1, activation='tanh', return_sequences=True,
                               dropout=self.dropout_rate, kernel_regularizer=regul_LX)(model_input)
            lstm_layer2 = LSTM(self.n_hidden_neurons_2, activation='tanh', dropout=self.dropout_rate, 
                               kernel_regularizer=regul_LX)(lstm_layer1)

            #concatenate external featrs on axis=1 with output of last LSTM layer (output of LSTM layer = sequence representation)
            merged_featrs = Concatenate(axis=1)([lstm_layer2, external_input]) 
            #Dense layer takes merged values as input:
            model_dense = Dense(self.n_preds,activation='linear')(merged_featrs)


            #compile:
            lstm_stacked_model = Model(inputs=[model_input, external_input], outputs=model_dense)
            lstm_stacked_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])

            # fit network
            history = lstm_stacked_model.fit([X_train, external_features_list[0]], y_train, epochs=self.n_epochs, 
                                             batch_size = self.n_batch_size, validation_data=([X_valid, external_features_list[1]],
                                                                                              y_valid), verbose=1, 
                                             shuffle = self.shuffle_flag, callbacks = callback_list) 



        if self.external_features_flag == False:
            print('Regular 2H-LSTM Model is created...')
            #create a regular model with single input:
            model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
            lstm_layer1 = LSTM(self.n_hidden_neurons_1, activation='tanh', return_sequences=True,
                               dropout=self.dropout_rate, kernel_regularizer=regul_LX)(model_input)
            lstm_layer2 = LSTM(self.n_hidden_neurons_2, activation='tanh', dropout=self.dropout_rate, 
                               kernel_regularizer=regul_LX)(lstm_layer1)
            model_dense = Dense(self.n_preds,activation='linear')(lstm_layer2)

            #compile:
            lstm_stacked_model = Model(inputs=[model_input], outputs=model_dense)
            lstm_stacked_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])

            # fit network
            history = lstm_stacked_model.fit(X_train, y_train, epochs=self.n_epochs, batch_size = self.n_batch_size, 
                                             validation_data=(X_valid, y_valid), verbose=1, shuffle = self.shuffle_flag, 
                                             callbacks = callback_list) 
        
        
        #assign/"store" model, history & model_name:
        self.prediction_model = lstm_stacked_model 
        self.model_name = model_name
        self.training_history = history

        return history, lstm_stacked_model
    

    
        

    #function to preprocess data, fit models and get predictions for valid & test set "automatically" 
    def create_full_pred_model(self, multivar_ts_series, start_train_year, last_train_set_year, 
                               start_validation_set_year, start_test_set_year, model_name, 
                               end_validation_set_year=None, end_test_set_year=None, get_preds_flag = True,
                               verbose=0):   
        
        
        '''
        creates full model "automatically": fits model on given data, makes prediction for following two years based on given input years
        '''
        
        
        
        if verbose == 1:
            print('start_validation_set_year ', start_validation_set_year)
            print('start_test_set_year ', start_test_set_year)
            print('end_validation_set_year ', end_validation_set_year)
            print('end_test_set_year ', end_test_set_year)
            
        
        
        #1)
        '''#call function to get preprocessed & reshaped data for model (differencing + scaling is applied 
                + lagged features are returned):   '''   
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, target_features_list = self.generate_data(
                                                                                        multivar_ts_series,
                                                                                        start_train_year,
                                                                                        last_train_set_year,
                                                                                        start_validation_set_year,
                                                                                        start_test_set_year, 
                                                                                        end_validation_set_year=end_validation_set_year,
                                                                                        end_test_set_year=end_test_set_year,
                                                                                        verbose=verbose)

        '''#Note: so far no approach available to use seasonal lagged features as input through an additional feature vector!! 
        --> quite unintuitive how to concat vector for each area '''


        #2)
        #create models:
        #create LSTM model:

        # create LSTM model:
        if self.stacked_flag == True:
            print('create stacked LSTM 2 layer non-stateful model:') 
            history_model, prediction_model = self.create_model_2H(X_train, y_train, X_valid, y_valid, 
                                                                   target_features_list, model_name)

        #create non-stacked model:
        else:
            print('create non-stateful LSTM model single layer:') 
            history_model, prediction_model = self.create_model_1H(X_train, y_train, X_valid, y_valid, 
                                                                   target_features_list, model_name)

        
        if get_preds_flag == True:
            #set flag: --> necessary to invert differencing correctly for multivariate time series
            multivariate_flag = True

            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year

            #prediction results validation data:
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.seasonal_lags_flag, target_features_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_validation_set_year, end_validation_set_year, True,
                                                                              scaler_list, self.standardizing_flag, self.scale_range, 
                                                                              prediction_model, multivar_ts_series, 
                                                                              'results_{}'.format(start_validation_set_year), verbose)

            #prediction results test data:
            if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
 
            if end_test_set_year == None:
                end_test_set_year = start_test_set_year

            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.seasonal_lags_flag, target_features_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_test_set_year, end_test_set_year, valid_flag,
                                                                              scaler_list, self.standardizing_flag, self.scale_range, 
                                                                              prediction_model, multivar_ts_series, 
                                                                              'results_{}'.format(start_test_set_year), verbose)



            return (history_model, prediction_model, predictions_results, model_name, multivar_ts_series, 
                    validation_results, rmse_results_test, rmse_results_valid)

        else:
            print('## Only training history & prediction model are returned')
            return (history_model, prediction_model)
           

    
    
    def retrain_model(self, multivar_series, start_train_year, last_train_set_year, start_validation_set_year, 
                      start_test_set_year, model_name, end_validation_set_year=None, 
                      end_test_set_year=None, n_epochs = 150, n_batch_size = 512, 
                      seasonal_lags_flag = False, external_features_flag = False, 
                      use_features_per_lag_flag = False, overwrite_params = False, 
                      get_preds_flag = True, verbose=0):
        
        '''
        #function creates new model and discards existing one --> function only calls "create_full_pred_model()"
        '''
        
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
            self.seasonal_lags_flag = seasonal_lags_flag
            self.external_features_flag = external_features_flag
            self.use_features_per_lag_flag = use_features_per_lag_flag
        

        print('## New Model is created, old model is discarded..')

        #create new MLP model:
        results_i = self.create_full_pred_model(multivar_series, start_train_year, last_train_set_year, 
                                                start_validation_set_year, start_test_set_year, model_name, 
                                                end_validation_set_year=end_validation_set_year, 
                                                end_test_set_year=end_test_set_year, get_preds_flag=get_preds_flag,                                        
                                                verbose=verbose)

        #returns results as tuple:
        return results_i

    
    

    
    def update_model_weights(self, multivar_series, start_train_year, last_train_set_year, start_validation_set_year, 
                             start_test_set_year, model_name, end_validation_set_year=None, end_test_set_year=None, 
                             n_epochs = 150, n_batch_size = 512, model_to_update = None, seasonal_lags_flag = False,
                             external_features_flag = False, use_features_per_lag_flag = False, 
                             overwrite_params = False, get_preds_flag = True, verbose=0):
        
        '''
        #function updates weights of exisiting model by fitting model on new data
        '''

        
        print('## Existing Model is updated..')
        
       
        
        if verbose == 1:
            print('start_validation_set_year ', start_validation_set_year)
            print('start_test_set_year ', start_test_set_year)
            print('end_validation_set_year ', end_validation_set_year)
            print('end_test_set_year ', end_test_set_year)
            
            
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
            self.seasonal_lags_flag = seasonal_lags_flag
            self.external_features_flag = external_features_flag
            self.use_features_per_lag_flag = use_features_per_lag_flag

        # 1) create data for model to be updated: 
        '''#Note: Only X_train & y_train are actually needed for updating the weights. X_valid and X_test are directly 
                    used to get predictions for these years'''
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, target_features_list = self.generate_data(multivar_series,
                                                                                            start_train_year,
                                                                                            last_train_set_year,
                                                                                            start_validation_set_year, 
                                                                                            start_test_set_year,
                                                                                            end_validation_set_year=end_validation_set_year,
                                                                                            end_test_set_year=end_test_set_year,
                                                                                            verbose=verbose)

        
        '''#Note: so far no approach available to append seasonal lagged features for multivariate case!! 
            --> quite unintuitive how to concat vector for each area --> "seasonal_lags_flag" should be always set to "False" 
        '''
        
        # 2) access model to be updated:
        if model_to_update:
            print('loaded model is used..')
            #if there is an existing model saved on disk that should be updated, "load" model into class:
            '''# !!!! Note: "loaded" model must have identical params as the Class currently has !!'''
            self.prediction_model = model_to_update
            
        prediction_model = self.prediction_model
        
        '''NOTE: "callback list" currently not available for updating the model!! 
        (would require to copy all callbacks into this function...)'''
        
        #updating --> valid data not used for fitting:
        if self.external_features_flag == False:
            
            #compile model:
            print('compile model')
            if self.clip_norm == True:
                print('#Clipping Norm applied')
                adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0, clipnorm=1.0)
            else:
                adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0)
            
            prediction_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])
            #fit model:
            #Note: use the local variables "n_epochs" & "batch_size" 
            history = prediction_model.fit(X_train, y_train, epochs=n_epochs, batch_size = n_batch_size, 
                                 verbose=1, shuffle = self.shuffle_flag) 
            
            
        if self.external_features_flag == True:
            external_features_list = target_features_list
            #compile model:
            print('compile model')
            if self.clip_norm == True:
                print('#Clipping Norm applied')
                adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0, clipnorm=1.0)
            else:
                adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0)

            prediction_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])
            #fit model:
            history = prediction_model.fit([X_train, external_features_list[0]], y_train, epochs=n_epochs, 
                                             batch_size = n_batch_size, verbose=1, 
                                             shuffle = self.shuffle_flag) 
            
        
        #assign/"store" model, history & model_name:
        self.prediction_model = prediction_model 
        self.model_name = model_name
        self.training_history = history
        
        
        # 3) get predictions with updated model:
        if get_preds_flag == True:
            #set flag: --> necessary to invert differencing correctly for multivariate time series
            multivariate_flag = True

            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year

            #prediction results validation data:
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.seasonal_lags_flag, target_features_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_validation_set_year, end_validation_set_year, 
                                                                              True, scaler_list, self.standardizing_flag, 
                                                                              self.scale_range, prediction_model, multivar_series, 
                                                                              'results_{}'.format(start_validation_set_year), verbose)

            #prediction results test data:
            if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
 
            if end_test_set_year == None:
                end_test_set_year = start_test_set_year

            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.seasonal_lags_flag, target_features_list, 
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_test_set_year, end_test_set_year, valid_flag, 
                                                                              scaler_list, self.standardizing_flag, self.scale_range,
                                                                              prediction_model, multivar_series, 
                                                                              'results_{}'.format(start_test_set_year), verbose)


            #create results tuple with only prediction results & model: (history does not contain information about validation data!!)
            results_i = (history, prediction_model, predictions_results, model_name, multivar_series,
                         validation_results, rmse_results_test, rmse_results_valid)
        
        else:
            print('## Only training history & prediction model are returned')
            results_i = (history, prediction_model)
        
        #returns results as tuple:
        return results_i
    
    
    
    
    
    

