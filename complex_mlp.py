import pandas as pd
import numpy as np
 

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


    
    
class ComplexMLP(BaseDeepClass):
    
    
    def __init__(self, n_timesteps=168, n_preds = 1, standardizing_flag = True, scale_range = (-1,1), seasonal_lags_flag = True,
                 n_batch_size = 512, n_epochs = 150, learning_rate = 0.001, adaptive_learning_rate_flag = False, 
                 early_stopping_flag = False, shuffle_flag=True, dropout_rate = 0.3, 
                 regularizer_lambda1 = 0, regularizer_lambda2 = 0, clip_norm = True, 
                 n_hidden_neurons_1 = 128, n_hidden_neurons_2 = 32):
        
        
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
                        Note: for complexMLP actuals = DataFrame
                        
            >> model_name: name of model given during training
            
            >> training_history: most recent history of training    
            
        '''        

        self.n_timesteps = n_timesteps
        self.n_preds = n_preds
        
        self.standardizing_flag = standardizing_flag
        self.scale_range = scale_range
        self.seasonal_lags_flag = seasonal_lags_flag
        
        self.n_batch_size = n_batch_size 
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate 
        self.adaptive_learning_rate_flag = adaptive_learning_rate_flag
                        
        self.n_hidden_neurons_1 = n_hidden_neurons_1
        self.n_hidden_neurons_2 = n_hidden_neurons_2

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
                      "model_name" : self.model_name,
                      "prediction_model" : self.prediction_model,
                      "actuals" : self.actuals
                     }
                     
        return param_dict
    
  
   
    def load_model(self, model_to_load):
        
        '''
        function "loads" Keras model which was stored on disk into Class
        
        Note: this only works if params of Class are the same as params used to train "model_to_load"
        
        '''
        
        self.prediction_model = model_to_load

    

    
    def get_complex_MLP_data_dict(self, multivar_series, seasonal_lag_set, n_timesteps):
        
        '''
        #function creates dfs for each area with taxi requests and additional features: one-hot-encoding of area labels, 
        weekofday, lags...
        #dfs are stored in dict
        '''

        multivar_series_copy = multivar_series.copy()
        #store column labels of org. data:
        area_labels = list(multivar_series_copy.columns)

        # 1) encode area_labels:
        areas_encoded_dict = self.create_one_hot_encoded_areas(multivar_series_copy)

        # 2) append weekofday encoding, hourofday encoding, monthofyear encoding, lags features:
        #Note: lags have to be appended on last step, since we drop rows with no valid lags (NaNs)

        #for each area:
        for i in range(len(area_labels)):

            #append weekofday encoded features
            weekofday_dummies_area_i = self.get_day_of_week_features_one_hot_encoded(multivar_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             weekofday_dummies_area_i],axis=1)

            #append hourofday encoding
            hourofdays_encod_area_i = self.get_hour_of_day_features_sin_cos_encoded(multivar_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             hourofdays_encod_area_i],axis=1)

            #append monthofyear encoding
            monthofyear_encod_area_i = self.get_month_of_year_features_sin_cos_encoded(multivar_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             monthofyear_encod_area_i],axis=1)


        #append lagged (seasonal lags & sliding window) and get final dataset for each area with all features:
        #-> this way we already get a "supervised" dataset very efficiently
        self.seasonal_lags_flag = True
        for i in range(len(area_labels)):
            ts_all_featrs = self.create_supervised_data_single_ts(areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                  self.n_timesteps, self.seasonal_lags_flag, seasonal_lag_set)

            #assign final dataset of each area to dict:
            areas_encoded_dict['area{}'.format(area_labels[i])] = ts_all_featrs


        return areas_encoded_dict
    
    
    
    
    
    
    def generate_data(self, multivar_series, start_train_year, last_train_set_year, 
                      start_validation_set_year, start_test_set_year, 
                      end_validation_set_year=None, end_test_set_year=None, verbose=0):               

        
        if verbose == 1:
            print('generate data..')
            print('start_train_year: ', start_train_year)
            print('last_train_set_year: ', last_train_set_year)
            
            print('start_validation_set_year: ', start_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('end_test_set_year: ', end_test_set_year)
        
        
        # 1) apply differencing
        ts_diff = self.differencing(multivar_series,1)
        #print(ts_diff)

        # 2) get df for each area with encoded features appended:
        lag_set = [168,336,504,672]
        areas_encoded_dict = self.get_complex_MLP_data_dict(ts_diff, lag_set, self.n_timesteps)


        # 3) get Train/Test-Split for each area & scale data:
        
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        
        if verbose == 1:
            print('# adjusted dates..')
            print('start_train_year: ', start_train_year)
            print('last_train_set_year: ', last_train_set_year)
            
            print('start_validation_set_year: ', start_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('end_test_set_year: ', end_test_set_year)
            
        
        #create dict to store results:
        supervised_data_dict = {}
        #create list to store scaler of each area:
        scaler_list = []
        #for each area create tain/test split & scale data & append to dict:
        for key in areas_encoded_dict:
            #get train/validation/test split:      
            ts_train = areas_encoded_dict[key].loc[start_train_year:last_train_set_year] 
            ts_test = areas_encoded_dict[key].loc[start_test_set_year:end_test_set_year]             
            ts_valid = areas_encoded_dict[key].loc[start_validation_set_year:end_validation_set_year]
            
                

            '''#Note: create final valid_set by adding last window of ts_train is not needed 
                        since we created supervised data with shift() function'''

            #slice first column of each df & (taxi request and lagged features (stored at last columns)) -> need to be scaled:
            ts_train_slice = pd.concat([pd.DataFrame(ts_train.iloc[:,0]),ts_train.iloc[:,-(self.n_timesteps+len(lag_set)):]],axis=1)
            ts_valid_slice = pd.concat([pd.DataFrame(ts_valid.iloc[:,0]),ts_valid.iloc[:,-(self.n_timesteps+len(lag_set)):]],axis=1)         
            ts_test_slice = pd.concat([pd.DataFrame(ts_test.iloc[:,0]),ts_test.iloc[:,-(self.n_timesteps+len(lag_set)):]],axis=1)

            #create arrays for scaler:
            ts_train_slice = ts_train_slice.values
            ts_valid_slice = ts_valid_slice.values
            ts_test_slice = ts_test_slice.values

            # Apply scaling::
            #Note: data_scaler() returns dfs!!  
            if verbose == 1:
                print('Data is scaled...')
            scaler, train_scaled, valid_scaled, test_scaled = self.data_scaler(ts_train_slice, ts_valid_slice, ts_test_slice,
                                                                               self.scale_range, self.standardizing_flag, verbose)

            #restore index: --> necessary to concat dfs correctly
            train_scaled.index = ts_train.index
            valid_scaled.index = ts_valid.index
            test_scaled.index = ts_test.index

            #concat scaled data and remaining features:
            train_scaled = pd.concat([train_scaled,ts_train.iloc[:,1:-(self.n_timesteps+len(lag_set))]],axis=1)
            valid_scaled = pd.concat([valid_scaled,ts_valid.iloc[:,1:-(self.n_timesteps+len(lag_set))]],axis=1)
            test_scaled = pd.concat([test_scaled,ts_test.iloc[:,1:-(self.n_timesteps+len(lag_set))]],axis=1)

            #append X,y pairs to dict:        
            #since we already created "supervised" data by including lags in df, we only have to slice df to receive X and y:
            X_train = train_scaled.iloc[:,1:].values
            X_valid  = valid_scaled.iloc[:,1:].values                   
            X_test = test_scaled.iloc[:,1:].values

            y_train = train_scaled.iloc[:,0].values       
            y_valid = valid_scaled.iloc[:,0].values
            y_test = test_scaled.iloc[:,0].values 

            #append results to dict:
            supervised_data_dict[key] = []
            supervised_data_dict[key].append(X_train)
            supervised_data_dict[key].append(y_train)
            supervised_data_dict[key].append(X_valid)
            supervised_data_dict[key].append(y_valid)
            supervised_data_dict[key].append(X_test)
            supervised_data_dict[key].append(y_test)

            #append each scaler to list:
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

        #create dict to store results:
        key_list = list(supervised_data_dict.keys())

        #fill arrays with entries of first area:
        X_train, y_train = supervised_data_dict[key_list[0]][0], supervised_data_dict[key_list[0]][1]
        X_valid, y_valid = supervised_data_dict[key_list[0]][2], supervised_data_dict[key_list[0]][3]
        X_test, y_test = supervised_data_dict[key_list[0]][4], supervised_data_dict[key_list[0]][5]

        for i in range(1,len(key_list)):
            X_train = np.concatenate((X_train,supervised_data_dict[key_list[i]][0]),axis=0)
            X_valid = np.concatenate((X_valid,supervised_data_dict[key_list[i]][2]),axis=0)
            X_test = np.concatenate((X_test,supervised_data_dict[key_list[i]][4]),axis=0)
            y_train = np.concatenate((y_train,supervised_data_dict[key_list[i]][1]),axis=0)
            y_valid = np.concatenate((y_valid,supervised_data_dict[key_list[i]][3]),axis=0)
            y_test = np.concatenate((y_test,supervised_data_dict[key_list[i]][5]),axis=0)

        if verbose == 1:
            print('final concatenated shape of X_train : ', X_train.shape)
            
            
        #assign/"store" actuals    
        self.actuals = multivar_series
        

        #Note: necessary to return scaler: we need scaler to invert scaling for predictions. 
        return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, supervised_data_dict



    
    
    #function that creates data for model that are only used for prediction task NOT training:
    def generate_data_get_predictions(self, multivar_series, start_train_year, last_train_set_year, 
                                      start_validation_set_year, start_test_set_year, 
                                      end_validation_set_year=None, end_test_set_year=None, verbose=0):

        #call function to create data:
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, supervised_data_dict = self.generate_data(multivar_series,
                                                                                start_train_year,
                                                                                last_train_set_year,
                                                                                start_validation_set_year,
                                                                                start_test_set_year, 
                                                                                end_validation_set_year=end_validation_set_year,
                                                                                end_test_set_year=end_test_set_year,
                                                                                verbose=verbose)            

        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        
        #get predictions for validation data:
        valid_flag = True
        validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.n_preds, scaler_list, self.standardizing_flag,
                                                                          self.scale_range, start_validation_set_year, 
                                                                          end_validation_set_year, valid_flag, 
                                                                          self.prediction_model, multivar_series, 
                                                                          supervised_data_dict, 
                                                                          'results_{}'.format(start_validation_set_year), verbose)
        
        #prediction results test data:
        valid_flag = False
        predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.n_preds, scaler_list, self.standardizing_flag,
                                                                          self.scale_range, start_test_set_year, 
                                                                          end_test_set_year, valid_flag, 
                                                                          self.prediction_model, multivar_series, 
                                                                          supervised_data_dict, 'results_{}'.format(start_test_set_year),
                                                                          verbose)


        return validation_results, predictions_results, rmse_results_valid, rmse_results_test

    
    
    
    
    
    #compile & fit Stacked MLP Model:
    def create_model(self, X_train, y_train, X_valid, y_valid, model_name, 
                    save_best_model_per_iteration = False, model_save_path=None):

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

        if save_best_model_per_iteration == True:
            #set path to store best model:
            Save_PATH = (model_save_path)

            '''#create Callback to store best model during training (until training is stopped if EarlyStopping is applied) 
                        (-> model at earlystopping, and best model before earlystopping is applied might differ):'''
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


        #definition of model: 

        #define model input tensor of sequences -> for MLP only 2D Input needed:
        model_input = Input(shape=(X_train.shape[1],))

        #create layers of MLP:
        dense1 = Dense(self.n_hidden_neurons_1, activation='relu', kernel_regularizer=regul_LX)(model_input)
        #add dropout-layer since Dense-Layer does not use "dropout" parameter
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense2 = Dense(self.n_hidden_neurons_2, activation='relu', kernel_regularizer=regul_LX)(dense1)     
        dense2 = Dropout(self.dropout_rate)(dense2)
        #last Dense layer:
        model_dense = Dense(self.n_preds, activation='linear')(dense2)

        #compile:
        mlp_model = Model(inputs=[model_input], outputs=model_dense)
        mlp_model.compile(optimizer=adam_opt, loss='mse', metrics=['mae'])

        # fit network
        history = mlp_model.fit([X_train], y_train, epochs=self.n_epochs, batch_size = self.n_batch_size, 
                                validation_data=([X_valid], y_valid), verbose=1, shuffle = self.shuffle_flag, 
                                callbacks = callback_list) 


        #assign/"store" model, history & model_name:
        self.prediction_model = mlp_model 
        self.model_name = model_name
        self.training_history = history
        
        return history, mlp_model

    
  
    

    
    #get predictions for complex MLP model (assuming scaling + differencing was applied on dataset):
    def get_preds_non_state(self, X_test, n_preds, scaler_list, standardizing_flag, scale_range, start_year_to_invert, 
                            end_year_to_invert, valid_flag, model, original_complete_dataset, supervised_data_dict_all_areas, model_name, verbose=0):

        
        #set years correctly:
        if end_year_to_invert == None:
            end_year_to_invert = start_year_to_invert
        
        
        # 1) get predictions based on sequence input & external input:
        yhat_s = model.predict([X_test], verbose=1)

        # 2) set index to select either validation or test set:
        if valid_flag == True:
            index_to_access = 2
        else:
            index_to_access = 4

        # 3) rescale predictions for each area according to individual scaler:
        yhat_rescaled_all_list = []

        #get keys of dict:
        key_list = list(supervised_data_dict_all_areas.keys())

        #prepare indices to access and store rescaled predictions:   
        start_idx = 0
        end_idx = 0

        #rescale predictions for each area:
        for i in range(len(key_list)):
            #set parameters to access correct number of values of each area:
            end_idx += supervised_data_dict_all_areas[key_list[i]][index_to_access].shape[0]
            number_of_samples = supervised_data_dict_all_areas[key_list[i]][index_to_access].shape[0]
            '''#access yhat_s at right index for each area -> since X_test contains inputs of each area underneath each other
                 concatenated on axis=0, we only need to access the right index..'''
            predictions_to_access = yhat_s[start_idx:end_idx,0] 
            #reshape data for scaler:
            predictions_to_access = predictions_to_access.reshape(number_of_samples,1)
            #call function to rescale predictions: / set n_preds to "1" for scaler
            yhat_rescaled_area_i = self.invert_data_scaler(scaler_list[i], predictions_to_access, standardizing_flag, scale_range, 1) 
            
            #append results:
            yhat_rescaled_all_list.append(yhat_rescaled_area_i)

            #update index:
            start_idx = end_idx

        '''#create numpy_array based on yhat_rescaled_all_list: 
            (this way we have all rescaled predictions for each area in one big numpy array -> each column equals an area)'''
        yhat_rescaled_all = yhat_rescaled_all_list[0]
        for i in range(1,len(yhat_rescaled_all_list)):
            yhat_rescaled_all = np.concatenate((yhat_rescaled_all,yhat_rescaled_all_list[i]),axis=1)


        # 4) invert differencing for each area:
        predictions_all = self.invert_differencing(original_complete_dataset, yhat_rescaled_all, start_year_to_invert, end_year_to_invert)
        
        if verbose == 1:
            print('predictions preview:')
            print(predictions_all.head(3))


        #get rmse:
        rmse_per_ts = []
        for u in range(predictions_all.shape[1]):
            rmse_single_ts = np.sqrt(mean_squared_error(original_complete_dataset.loc[start_year_to_invert:end_year_to_invert].iloc[:,u],
                                                        predictions_all.iloc[:,u]))
            rmse_per_ts.append(rmse_single_ts)
            
            if verbose == 1:
                print('RMSE per TS {} : model: {} : {}'.format(u, model_name, rmse_per_ts[u]))

        #get average of all rmses
        total_rmse = np.mean(rmse_per_ts)
        
        if verbose == 1:
            print('Avg.RMSE for complex MLP model {} : {}'.format(model_name, total_rmse))


        #return RMSE results:
        rmse_results = []
        rmse_results.append(total_rmse)
        rmse_results.append(rmse_per_ts)


        return predictions_all, rmse_results

  





    #function to preprocess data, fit model and get predictions for valid & test set "automatically" 
    def create_full_pred_model(self, multivar_series, start_train_year, last_train_set_year, 
                               start_validation_set_year, start_test_set_year, model_name, 
                               end_validation_set_year=None, end_test_set_year=None, get_preds_flag = True,
                               save_best_model_per_iteration = False, model_save_path=None,
                               verbose = 0):
        
        #1)
        #get data for model:
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, supervised_data_dict = self.generate_data(multivar_series,
                                                                                    start_train_year,
                                                                                    last_train_set_year,
                                                                                    start_validation_set_year,
                                                                                    start_test_set_year, 
                                                                                    end_validation_set_year=end_validation_set_year,
                                                                                    end_test_set_year=end_test_set_year,
                                                                                    verbose=verbose)                
        #2)
        #create model:
        print('create MLP Model:')       
        history_model, prediction_model = self.create_model(X_train, y_train, X_valid, y_valid, model_name,
                                                            save_best_model_per_iteration = save_best_model_per_iteration, 
                                                            model_save_path=model_save_path)
        
        
        #3) get preds
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
            
        
        if get_preds_flag == True:
            #get predictions for validation data:
            valid_flag = True
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.n_preds, scaler_list, self.standardizing_flag,
                                                                              self.scale_range, start_validation_set_year, 
                                                                              end_validation_set_year, valid_flag, 
                                                                              prediction_model, multivar_series, supervised_data_dict, 
                                                                              'results_{}'.format(start_validation_set_year), verbose)

            #prediction results test data:
            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.n_preds, scaler_list, self.standardizing_flag,
                                                                              self.scale_range, start_test_set_year, 
                                                                              end_test_set_year, valid_flag, 
                                                                              prediction_model, multivar_series, supervised_data_dict,
                                                                              'results_{}'.format(start_test_set_year), verbose)


            return (history_model, prediction_model, predictions_results, model_name, multivar_series, validation_results,
                    rmse_results_test, rmse_results_valid)

        else:
            print('## Only training history & model are returned')
            return (history_model, prediction_model)



    
    
    def retrain_model(self, multivar_series, start_train_year, last_train_set_year, 
                      start_validation_set_year, start_test_set_year, model_name, 
                      end_validation_set_year=None, end_test_set_year=None, 
                      n_epochs = 150, n_batch_size = 512, overwrite_params = False, 
                      get_preds_flag=True, save_best_model_per_iteration = False, model_save_path=None, 
                      verbose=0):
        
                     
        '''
        #function creates new model and discards existing one --> function only calls "create_full_pred_model()"
        n_epochs & batchsize can be set if "overwrite_params" = True
        '''
        
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
            
        
        print('## New Model is created, old model is discarded..')
               
        #create new MLP model:
        results_i = self.create_full_pred_model(multivar_series, start_train_year, last_train_set_year,
                                                start_validation_set_year, start_test_set_year, model_name, 
                                                end_validation_set_year=end_validation_set_year, 
                                                end_test_set_year=end_test_set_year, get_preds_flag = get_preds_flag,
                                                save_best_model_per_iteration = save_best_model_per_iteration, 
                                                model_save_path = model_save_path, 
                                                verbose=verbose)
        
        
        #returns results as tuple:
        return results_i

    
    

    
    def update_model_weights(self, multivar_series, start_train_year, last_train_set_year, start_validation_set_year,
                             start_test_set_year, model_name, end_validation_set_year=None, end_test_set_year=None, 
                             n_epochs = 150, n_batch_size = 512, model_to_update = None, overwrite_params = False,
                             get_preds_flag = True, verbose=0):
        
        '''
        #function updates weights of exisiting model by fitting model on new data
        '''

        
        print('## Existing Model is updated..')
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
        
        # 1) create data for model to be updated: 
        '''#Note: Only X_train & y_train are actually needed for updating the weights. 
                    X_valid and X_test are directly used to get predictions for these years'''
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, supervised_data_dict = self.generate_data(multivar_series,
                                                                                        start_train_year,
                                                                                        last_train_set_year, 
                                                                                        start_validation_set_year,
                                                                                        start_test_set_year, 
                                                                                        end_validation_set_year=end_validation_set_year,
                                                                                        end_test_set_year=end_test_set_year,
                                                                                        verbose=verbose)                


        # 2) access model to be updated:
        if model_to_update:
            print('loaded model is used..')
            #if there is an existing model saved on disk that should be updated, "load" model into class:
            # !!!! Note: "loaded" model must have identical params as the Class currently has !!
            self.prediction_model = model_to_update
            
        prediction_model = self.prediction_model
        '''NOTE: "callback list" currently not available for updating the model!! 
                    (would require to copy all callbacks into this function...)'''
        
        
        #compile model:
        #define optimizer for model:
        if self.clip_norm == True:
            print('#Clipping Norm applied')
            adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0, clipnorm=1.0)
        else:
            adam_opt = optimizers.Adam(lr=self.learning_rate, decay=0.0)
            
        prediction_model.compile(optimizer=adam_opt, loss='mse', metrics=['mae'])
        
        #updating --> valid data not used for fitting:
        #fit model:
        history = prediction_model.fit(X_train, y_train, epochs=n_epochs, batch_size = n_batch_size, verbose=1,
                             shuffle = self.shuffle_flag) 
        
        #assign/"store" model, history & model_name:
        self.prediction_model = prediction_model 
        self.model_name = model_name
        self.training_history = history
        
        
        # 3) get predictions with updated model:
        
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        
        if get_preds_flag == True:
            #get predictions for validation data:
            valid_flag = True
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.n_preds, scaler_list, self.standardizing_flag,
                                                                              self.scale_range, start_validation_set_year, 
                                                                              end_validation_set_year, valid_flag, 
                                                                              prediction_model, multivar_series, supervised_data_dict,
                                                                              'results_{}'.format(start_validation_set_year), verbose)

            #prediction results test data:
            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.n_preds, scaler_list, self.standardizing_flag,
                                                                              self.scale_range, start_test_set_year, 
                                                                              end_test_set_year, valid_flag,
                                                                              prediction_model, multivar_series, supervised_data_dict,
                                                                              'results_{}'.format(start_test_set_year), verbose)


            #create results tuple with only prediction results & model: (history does not contain information about validation data!!)
            results_i = (history, prediction_model, predictions_results, model_name, multivar_series,
                         validation_results, rmse_results_test, rmse_results_valid)

        
        else:
            print('## Only training history & model are returned')
            results_i = (history, prediction_model)
        
        #returns results as tuple:
        return results_i
    
    
    

    







    
    

