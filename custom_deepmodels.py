
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



class SingleLSTM(BaseDeepClass):
    
    
    def __init__(self, n_timesteps=168, n_preds = 1, standardizing_flag = False, scale_range = (-1,1), seasonal_lags_flag = False,
                 external_features_flag = False, n_batch_size = 512, n_epochs = 150, learning_rate = 0.001, 
                 adaptive_learning_rate_flag = False, early_stopping_flag = False, shuffle_flag=True, dropout_rate = 0.3, 
                 regularizer_lambda1 = 0, regularizer_lambda2 = 0, clip_norm = True, 
                 n_hidden_neurons_1 = 128, n_hidden_neurons_2 = 32, stacked_flag=True):
        
        
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
                                    
            >> external_features_flag: if set to "True" assumes features are available which are then
                                       incorporated into model as additional feature vector 
                                       through a dense layer
                                       
                                       -> added as extra flag since there might be other features in future implementations 
                                       apart from seasonal lags
                                       
                                       e.g: seasonal_lags_flag was set to "True" then seasonal lags are used 
                                            as additional feature vector
        
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
                        Note: for SingleLSTM actuals = pd.Series()
                    
            >> model_name: name of model given during training
            
            >> training_history: most recent history of training
            
        '''
        

        self.n_timesteps = n_timesteps
        self.n_preds = n_preds
        
        self.standardizing_flag = standardizing_flag
        self.scale_range = scale_range
        self.seasonal_lags_flag = seasonal_lags_flag
        self.external_features_flag = external_features_flag
        
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



        
        
    def generate_data(self, ts_series, start_train_year, last_train_set_year, start_validation_set_year, 
                      start_test_set_year, end_validation_set_year=None, end_test_set_year=None, verbose=0):
        
        '''
        function creastes features and splits data for training & returns reshaped data
        
        Note: end_validation_set_year & end_test_set_year must be specified if not a whole year (e.g. all observations in "2011"), but some months should be selected ('2011-02':'2011-04')
        
        Note: for single LSTM only "seasonal lags" (up to 4 week lags based on target) and "regular lags" are available as "features"
        Note: "seasonal lags" can not be processed by LSTM as a sequence but have to be used as additional input 
        through a Dense Layer.. --> are returned seperately
        
               
        '''
                
        
        # 1) apply differencing
        ts_diff = self.differencing(ts_series,1)
        #print(ts_diff)

        # 2) create supervised data:
        seasonal_lag_set = [168,336,504,672]
        ts_diff = self.create_supervised_data_single_ts(ts_diff, self.n_timesteps, self.seasonal_lags_flag , seasonal_lag_set)
        print('ts diff shape: ', ts_diff.shape)


        # 3) get Train/Test-Split:
        print('Train/Test Split...')
        ts_train = ts_diff.loc[start_train_year:last_train_set_year]
        
        
        #get train/test Split:
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        ts_test = ts_diff.loc[start_test_set_year:end_test_set_year]
        
        #get train/validation Split:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
            
        ts_valid = ts_diff.loc[start_validation_set_year:end_validation_set_year]
        
        
        if verbose == 1:
            print('generate data..')
            print('start_validation_set_year: ', start_validation_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_test_set_year: ', end_test_set_year)
            

        # 4) scale data:   
        #create arrays for scaler:
        train_array = ts_train.values
        valid_array = ts_valid.values
        test_array = ts_test.values

        #Note: no reshaping necessary since each dataset contains multiple columns: actuals, lags, seasonal lags

        #check correct shape of arrays:
        #print('Train array before reshaping: ', train_array)
        #print('Train array shape: ', train_array.shape)


        # Apply scaling::
        print('Data is scaled...')
        #scale data:
        scaler, train_scaled, valid_scaled, test_scaled = self.data_scaler(train_array, valid_array, test_array, 
                                                                           self.scale_range, self.standardizing_flag, verbose)
        #print(train_scaled)

        #make sure type is float:
        train_scaled = train_scaled.astype('float64')
        valid_scaled = valid_scaled.astype('float64')
        test_scaled = test_scaled.astype('float64')


        # 5) get X, y data:        
        #get supervised data with lagged features:   
        if self.seasonal_lags_flag == True:
            #assign seasonal_lags to extra variable:
            X_train = train_scaled.iloc[:,1:(len(range(self.n_timesteps))+1)].values
            y_train = train_scaled.iloc[:,0].values
            seasonal_lags_train = train_scaled.iloc[:,(-len(seasonal_lag_set)):].values

            
            X_valid = valid_scaled.iloc[:,1:(len(range(self.n_timesteps))+1)].values                   
            y_valid = valid_scaled.iloc[:,0].values
            seasonal_lags_valid = valid_scaled.iloc[:,(-len(seasonal_lag_set)):].values
            
            X_test = seasonal_lags_test = test_scaled.iloc[:,1:(len(range(self.n_timesteps))+1)].values 
            y_test = test_scaled.iloc[:,0].values
            seasonal_lags_test = test_scaled.iloc[:,(-len(seasonal_lag_set)):].values
            
            
        else:
            X_train, y_train = train_scaled.iloc[:,1:].values, train_scaled.iloc[:,0].values
            X_valid, y_valid = valid_scaled.iloc[:,1:].values, valid_scaled.iloc[:,0].values                    
            X_test, y_test = test_scaled.iloc[:,1:].values, test_scaled.iloc[:,0].values 


        # 6) reshape data for models:
        #for LSTM models we need 3D tensors:
        print('Reshape data for LSTM model...')   
        #reshape data for model: (#samples , #timesteps, #features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.n_preds))
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], self.n_preds))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.n_preds))

        
        if verbose == 1:
            print('X_train shape before modeling: ', X_train.shape)
            print('X_valid shape before modeling: ', X_valid.shape)
            print('X_test shape before modeling: ', X_test.shape)
            print('y_train shape before modeling: ', y_train.shape)
            print('y_valid shape before modeling: ', y_valid.shape)
            print('y_test shape before modeling: ', y_test.shape)


        # 7) reshape lagged features if lagged features are used as external input (not in sequence but as additional feature vector):
        if self.seasonal_lags_flag == True:
            #reshape lagged features:
            #we only need 2D shape:
            seasonal_lags_train = seasonal_lags_train.reshape((seasonal_lags_train.shape[0], seasonal_lags_train.shape[1]))
            seasonal_lags_valid = seasonal_lags_valid.reshape((seasonal_lags_valid.shape[0], seasonal_lags_valid.shape[1]))        
            seasonal_lags_test = seasonal_lags_test.reshape((seasonal_lags_test.shape[0], seasonal_lags_test.shape[1]))
            
            if verbose == 1:
                print('lagged_target_features_train shape before modeling: ', seasonal_lags_train.shape)

            #create list of lagged_input_features:
            seasonal_lags_list = [seasonal_lags_train, seasonal_lags_valid, seasonal_lags_test]


        else:
            #set flag to False and create empty list:
            seasonal_lags_list = []

        #store scaler in list:
        scaler_list = []
        scaler_list.append(scaler)

        #double check scaler type:
        if verbose == 1:
            print('scaler type: ', scaler_list[0])
            
            
        #assign/"store" actuals:
        self.actuals = ts_series

        #Note: necessary to return scaler: we need scaler to invert scaling for predictions. 
        return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, seasonal_lags_list



    

    def generate_data_get_predictions(self, ts_series_single_area, start_train_year, last_train_set_year, 
                                      start_validation_set_year, start_test_set_year, end_validation_set_year=None, 
                                      end_test_set_year=None, verbose=0):
        
        '''
        function that creates data for LSTM model that are only used for prediction task NOT training:
        
        '''
        
        #assign/"store" actuals:
        self.actuals = ts_series_single_area
                

        
        #call function to create data:
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, seasonal_lags_list = self.generate_data(
                                                                                ts_series_single_area, start_train_year,
                                                                                last_train_set_year,
                                                                                start_validation_set_year, 
                                                                                start_test_set_year, 
                                                                                end_validation_set_year=end_validation_set_year, 
                                                                                end_test_set_year=end_test_set_year,
                                                                                verbose=verbose)                                                                                          
                                                                                                
        #set flag:
        multivariate_flag = False
        
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year

        #prediction results validation data:
        validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.seasonal_lags_flag, seasonal_lags_list,
                                                                          multivariate_flag, self.n_preds, 
                                                                          start_validation_set_year, end_validation_set_year, True,
                                                                          scaler_list, self.standardizing_flag, self.scale_range, 
                                                                          self.prediction_model, ts_series_single_area, 
                                                                          'results_{}'.format(start_validation_set_year), verbose)

        #prediction results test data: 
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
            
        valid_flag = False
        predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.seasonal_lags_flag, seasonal_lags_list,
                                                                          multivariate_flag, self.n_preds, 
                                                                          start_test_set_year, end_test_set_year, valid_flag,
                                                                          scaler_list, self.standardizing_flag, self.scale_range,
                                                                          self.prediction_model, ts_series_single_area, 
                                                                          'results_{}'.format(start_test_set_year), verbose)


        return validation_results, predictions_results, rmse_results_valid, rmse_results_test



    

    
    # compiles and fits static (non-stateful) 1H LSTM Network
    def create_model_1H(self, X_train, y_train, X_valid, y_valid, external_features_list, model_name, 
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

        if save_best_model_per_iteration == True:
            #set path to store best model:
            Save_PATH = (model_save_path)
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
    def create_model_2H(self, X_train, y_train, X_valid, y_valid, external_features_list, model_name,
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
    def create_full_pred_model(self, ts_series_single_area, start_train_year, last_train_set_year, 
                               start_validation_set_year, start_test_set_year, model_name, 
                               end_validation_set_year = None, end_test_set_year = None, 
                               get_preds_flag = True, save_best_model_per_iteration = False, model_save_path=None, 
                               verbose=0):          
        
        
        
        if verbose == 1:
            print('years selected:')
            print('start_validation_set_year ', start_validation_set_year)
            print('start_test_set_year ', start_test_set_year)
            print('end_validation_set_year ', end_validation_set_year)
            print('end_test_set_year ', end_test_set_year)
            print('##')
        
        #1)
        #call function to get preprocessed & reshaped data for model (differencing + scaling is applied 
        #      + lagged features are returned):      
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, target_features_list = self.generate_data(
                                                                                ts_series_single_area, start_train_year,
                                                                                last_train_set_year,
                                                                                start_validation_set_year, 
                                                                                start_test_set_year, 
                                                                                end_validation_set_year=end_validation_set_year, 
                                                                                end_test_set_year=end_test_set_year,
                                                                                verbose=verbose)                   

        #2)
        #create models:
        #create LSTM model:

        # create LSTM model:
        if self.stacked_flag == True:
            print('create stacked LSTM 2 layer non-stateful model:') 
            history_model, prediction_model = self.create_model_2H(X_train, y_train, X_valid, y_valid, 
                                                                   target_features_list, model_name,
                                                                   save_best_model_per_iteration = save_best_model_per_iteration, 
                                                                   model_save_path=model_save_path)

        #create non-stacked model:
        else:
            print('create non-stateful LSTM model single layer:') 
            history_model, prediction_model = self.create_model_1H(X_train, y_train, X_valid, y_valid, 
                                                                   target_features_list, model_name,                                                                                                                                            
                                                                   save_best_model_per_iteration = save_best_model_per_iteration, 
                                                                   model_save_path=model_save_path)

        #3) get preds:
        if get_preds_flag == True:
            #set flag:
            multivariate_flag = False

            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year

            #prediction results validation data:
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.seasonal_lags_flag, target_features_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_validation_set_year, end_validation_set_year, True,
                                                                              scaler_list, self.standardizing_flag, self.scale_range, 
                                                                              self.prediction_model, ts_series_single_area, 
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
                                                                              self.prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_test_set_year), verbose)




            return (history_model, prediction_model, predictions_results, model_name, ts_series_single_area, validation_results,
                    rmse_results_test, rmse_results_valid)

        else:
            print('## Only training history & model are returned')
            return (history_model, prediction_model)
  


    #function creates new model and discards existing one --> function only calls "create_full_pred_model()"
    def retrain_model(self, ts_series_single_area, start_train_year, last_train_set_year, start_validation_set_year, 
                      start_test_set_year, model_name, stacked_flag = True, end_validation_set_year = None, 
                      end_test_set_year = None, n_epochs = 150, n_batch_size = 512, seasonal_lags_flag = False,
                      external_features_flag = False, overwrite_params = False, get_preds_flag = True, 
                      save_best_model_per_iteration = False, model_save_path=None, verbose=0):
        
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
            self.seasonal_lags_flag = seasonal_lags_flag
            self.external_features_flag = external_features_flag
            
        
        print('## New Model is created, old model is discarded..')
        
        #reassign stacked flag if wanted
        self.stacked_flag = stacked_flag
        
        if verbose == 1:
            print('start_validation_set_year ', start_validation_set_year)
            print('start_test_set_year ', start_test_set_year)
            print('end_validation_set_year ', end_validation_set_year)
            print('end_test_set_year ', end_test_set_year)
        
        #create new LSTM model:
        results_i = self.create_full_pred_model(ts_series_single_area, start_train_year, last_train_set_year, 
                                                start_validation_set_year, start_test_set_year, model_name, 
                                                end_validation_set_year=end_validation_set_year, 
                                                end_test_set_year=end_test_set_year, get_preds_flag = get_preds_flag,
                                                save_best_model_per_iteration = save_best_model_per_iteration, 
                                                model_save_path=model_save_path, verbose=verbose)                                      

        #returns results as tuple:
        return results_i

    
    
    def update_model_weights(self, ts_series_single_area, start_train_year, last_train_set_year, start_validation_set_year, 
                             start_test_set_year, model_name, end_validation_set_year = None, end_test_set_year = None,
                             n_epochs = 150, n_batch_size = 512, model_to_update = None, 
                             seasonal_lags_flag = False, external_features_flag = False, overwrite_params = False,
                             get_preds_flag = True, verbose=0):
        
        print('## Existing Model is updated..')
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
            self.seasonal_lags_flag = seasonal_lags_flag
            self.external_features_flag = external_features_flag
        
        if verbose == 1:
            print('start_validation_set_year ', start_validation_set_year)
            print('start_test_set_year ', start_test_set_year)
            print('end_validation_set_year ', end_validation_set_year)
            print('end_test_set_year ', end_test_set_year)
            
        
        # 1) create data for model to be updated: 
        #Note: Only X_train & y_train are actually needed for updating the weights. 
        #      X_valid and X_test are directly used to get predictions for these years
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list, seasonal_lags_list = self.generate_data(
                                                                                ts_series_single_area, start_train_year,
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
        
        
        #updating --> valid data not used for fitting:        
        if self.external_features_flag == False:
            
            #define optimizer for model:
            if self.clip_norm == True:
                print('#Clipping Norm applied')
                adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0, clipnorm=1.0)
            else:
                adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0)
            
            #compile model:
            prediction_model.compile(optimizer=adam_opt, loss='mse',metrics=['mae'])
            
            #fit model:
            history = prediction_model.fit(X_train, y_train, epochs=n_epochs, batch_size = n_batch_size, verbose=1, 
                                 shuffle = self.shuffle_flag) 
        
        
        # fit network with external_features:
        if self.external_features_flag == True:
            
            external_features_list = seasonal_lags_list
            
            #define optimizer for model:
            if self.clip_norm == True:
                print('#Clipping Norm applied')
                adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0, clipnorm=1.0)
            else:
                adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0)
            
            #compile model:
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
            #set flag:
            multivariate_flag = False

            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year

            #prediction results validation data:
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, self.seasonal_lags_flag, seasonal_lags_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_validation_set_year, end_validation_set_year, True,
                                                                              scaler_list, self.standardizing_flag, self.scale_range, 
                                                                              self.prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_validation_set_year), verbose)


            #prediction results test data: 
            if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
            if end_test_set_year == None:
                end_test_set_year = start_test_set_year

            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, self.seasonal_lags_flag, seasonal_lags_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_test_set_year, end_test_set_year, valid_flag,
                                                                              scaler_list, self.standardizing_flag, self.scale_range,
                                                                              self.prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_test_set_year), verbose)



            #create results tuple with only prediction results & model: (history does not contain information about validation data!!)
            results_i = (history, prediction_model, predictions_results, model_name, ts_series_single_area,
                         validation_results, rmse_results_test, rmse_results_valid)
        
        
        else:
            print('## Only training history & model are returned')
            results_i = (history, prediction_model) 
        
        #returns results as tuple:
        return results_i
        
  


        


        
  


        

        

    
    
    
    
    
    