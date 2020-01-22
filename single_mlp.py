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







class SingleMLP(BaseDeepClass):
    
    
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
                        Note: for SingleMLP actuals = pd.Series()
                        
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

    
    
    
      
        
        
    def generate_data(self, ts_series, start_train_year, last_train_set_year, start_validation_set_year, start_test_set_year, 
                      end_validation_set_year=None, end_test_set_year=None, verbose=0):               
    
        '''
        function creates applies differencing, splits data based on given years and scales data based on training set. 
        The data sets can then be used for modeling & predictions
        
        '''
        
          
        if verbose == 1:
            print('generate data..')
            print('start_validation_set_year: ', start_validation_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_test_set_year: ', end_test_set_year)

        
        # 1) apply differencing
        ts_diff = self.differencing(ts_series,1)
        #print(ts_diff)


        # 2) create supervised data:
        seasonal_lag_set = [168,336,504,672]
        #Note: for simple_MLP seasonal_lags_flag is always True:
        self.seasonal_lags_flag = True
        #get lags and supervised shape:
        ts_diff = self.create_supervised_data_single_ts(ts_diff, self.n_timesteps, self.seasonal_lags_flag, seasonal_lag_set)
        
        #print('ts diff shape: ', ts_diff.shape)


        # 3) get Train/Test-Split:
        if verbose == 1:
            print('Train/Test Split...')
            
        #set years correctly:
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
            
        if verbose == 1:
            print('# dates adjusted:')
            print('start_validation_set_year: ', start_validation_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_test_set_year: ', end_test_set_year)
                    
        #split data:
        ts_train = ts_diff.loc[start_train_year:last_train_set_year]
        ts_test = ts_diff.loc[start_test_set_year:end_test_set_year]
        ts_valid = ts_diff.loc[start_validation_set_year:end_validation_set_year]


        # 4) scale data:   
        #create arrays for scaler:
        train_array = ts_train.values
        valid_array = ts_valid.values
        test_array = ts_test.values

        #Note: no reshaping necessary since each dataset contains multiple columns: actuals, lags, seasonal lags..

        #check correct shape of arrays:
        #print('Train array before reshaping: ', train_array)
        #print('Train array shape: ', train_array.shape)


        # Apply scaling::
        if verbose == 1:
            print('Data is scaled...')
        scaler, train_scaled, valid_scaled, test_scaled = self.data_scaler(train_array, valid_array, test_array, 
                                                                           self.scale_range, self.standardizing_flag, verbose)
        #print(train_scaled)

        #make sure type is float:
        train_scaled = train_scaled.astype('float64')
        valid_scaled = valid_scaled.astype('float64')
        test_scaled = test_scaled.astype('float64')


        # 5) get X, y data:        
        #get supervised data with lagged features:       
        X_train, y_train = train_scaled.iloc[:,1:].values, train_scaled.iloc[:,0].values
        X_valid, y_valid = valid_scaled.iloc[:,1:].values, valid_scaled.iloc[:,0].values                    
        X_test, y_test = test_scaled.iloc[:,1:].values, test_scaled.iloc[:,0].values 


        # 6) reshape data for models:
        #for MLP models we need 2D tensors:
        print('Reshape data for MLP model...')   
        #reshape data for model: (#samples , #timesteps)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
        
        if verbose == 1:
            print('X_train shape before modeling: ', X_train.shape)
            print('X_valid shape before modeling: ', X_valid.shape)
            print('X_test shape before modeling: ', X_test.shape)
            print('y_train shape before modeling: ', y_train.shape)
            print('y_valid shape before modeling: ', y_valid.shape)
            print('y_test shape before modeling: ', y_test.shape)

        #store scaler of area:
        scaler_list = []
        scaler_list.append(scaler)

        #double check scaler type:
        if verbose == 1:
            print('scaler type: ', scaler_list[0])
            
            
        #assign/"store" actuals    
        self.actuals = ts_series

        #Note: necessary to return scaler: we need scaler to invert scaling for predictions. 
        return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list




    def generate_data_get_predictions(self, ts_series_single_area, start_validation_set_year, start_test_set_year, 
                                      end_validation_set_year=None, end_test_set_year=None, verbose = 0):
        
        
        '''
        function that creates data for multivar model that are only used for prediction task NOT training:
        '''      
        
        
        #call function to create data: (Note: for predictions only X_valid & X_test are needed..)
        self.seasonal_lags_flag = True
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list =  self.generate_data(ts_series_single_area, 
                                                                                start_train_year, 
                                                                                last_train_set_year,
                                                                                start_validation_set_year, 
                                                                                start_test_set_year,
                                                                                end_validation_set_year=end_validation_set_year,
                                                                                end_test_set_year=end_test_set_year,
                                                                                verbose=verbose)             

        #set some parameters to call get_preds_non_state() function properly:
        #necessary to set list to empty list since it is not used by simple MLP
        target_featrs_list = []
        
        '''#Note: necessary to set "additional_features_flag" to "False" --> the MLP model can incorporate seasonal lags directly! 
            -> not necessary to create an extra vector, therefore indicate "get_preds_non_state()" 
                that no extra vector is available'''
        additional_features_flag = False
        
        multivariate_flag = False
        
            
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
            
            
        #prediction results validation data:
        validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, additional_features_flag, target_featrs_list,
                                                                          multivariate_flag, self.n_preds, 
                                                                          start_validation_set_year, end_validation_set_year,
                                                                          True, scaler_list, self.standardizing_flag, 
                                                                          self.scale_range, self.prediction_model, 
                                                                          ts_series_single_area, 
                                                                          'results_{}'.format(start_validation_set_year), verbose)        

        #prediction results test data:              
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
            
        valid_flag = False
        predictions_results, rmse_results_test = self.get_preds_non_state(X_test, additional_features_flag, target_featrs_list,
                                                                          multivariate_flag, self.n_preds, start_test_set_year, 
                                                                          end_test_set_year, valid_flag,
                                                                          scaler_list, self.standardizing_flag, self.scale_range, 
                                                                          self.prediction_model, ts_series_single_area, 
                                                                          'results_{}'.format(start_test_set_year), verbose)

        
        return validation_results, predictions_results, rmse_results_valid, rmse_results_test


    
    
    
    #Stacked MLP Model:
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



    
    
    
    #function to preprocess data, fit models and get predictions for valid & test set "automatically" 
    def create_full_pred_model(self,ts_series_single_area, start_train_year, last_train_set_year, 
                               start_validation_set_year, start_test_set_year, model_name, 
                               end_validation_set_year=None, end_test_set_year=None, 
                               get_preds_flag = True, save_best_model_per_iteration = False, model_save_path=None, 
                               verbose=0):          
        
        
        #1)
        '''#call function to get preprocessed & reshaped data for model 
                (differencing + scaling is applied + lagged features are returned): '''     
        self.seasonal_lags_flag = True
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list =  self.generate_data(ts_series_single_area, 
                                                                                start_train_year,
                                                                                last_train_set_year,
                                                                                start_validation_set_year, 
                                                                                start_test_set_year,
                                                                                end_validation_set_year=end_validation_set_year,
                                                                                end_test_set_year=end_test_set_year,
                                                                                verbose=verbose)             

        #2)
        #create models:
        #create LSTM model:
        history_model, prediction_model = self.create_model(X_train, y_train, X_valid, y_valid, model_name,
                                                           save_best_model_per_iteration = save_best_model_per_iteration, 
                                                            model_save_path = model_save_path)

        #3) get preds:        
        if get_preds_flag == True:
            #set some parameters to call get_preds_non_state() function properly:
            #necessary to set list to empty list since it is not used by simple MLP
            target_featrs_list = []

            '''#Note: necessary to set "additional_features_flag" to "False" --> the MLP model can incorporate seasonal lags directly! 
                -> not necessary to create an extra vector, therefore indicate "get_preds_non_state()" 
                that no extra vector is available'''
            additional_features_flag = False

            multivariate_flag = False

            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year

            #prediction results validation data:
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, additional_features_flag, target_featrs_list,
                                                                              multivariate_flag, self.n_preds, start_validation_set_year,
                                                                              end_validation_set_year, True,
                                                                              scaler_list, self.standardizing_flag, self.scale_range,
                                                                              prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_validation_set_year), verbose)               

            #prediction results test data:
            if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
            
            if end_test_set_year == None:
                end_test_set_year = start_test_set_year

            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, additional_features_flag, target_featrs_list,
                                                                              multivariate_flag, self.n_preds, start_test_set_year,
                                                                              end_test_set_year, valid_flag,
                                                                              scaler_list, self.standardizing_flag, self.scale_range, 
                                                                              prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_test_set_year), verbose)


            return (history_model, prediction_model, predictions_results, model_name, ts_series_single_area,
                    validation_results, rmse_results_test, rmse_results_valid)
        
        else:
            print('## Only training history & model are returned')
            return (history_model, prediction_model)
    
     

        
    
    def retrain_model(self, ts_series_single_area, start_train_year, last_train_set_year, 
                      start_validation_set_year, start_test_set_year, model_name, 
                      end_validation_set_year=None, end_test_set_year=None, 
                      n_epochs = 150, n_batch_size = 512, overwrite_params = False, 
                      get_preds_flag = True, save_best_model_per_iteration = False, model_save_path=None, 
                      verbose=0):
        
        '''
        #function creates new model and discards existing one --> function only calls "create_full_pred_model()"
        '''
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_epochs = n_epochs
            self.n_batch_size = n_batch_size
            
        
        print('## New Model is created, old model is discarded..')
               
        #create new MLP model:
        results_i = self.create_full_pred_model(ts_series_single_area, start_train_year, last_train_set_year, 
                                                start_validation_set_year, start_test_set_year, model_name, 
                                                end_validation_set_year=end_validation_set_year, 
                                                end_test_set_year=end_test_set_year, 
                                                get_preds_flag = get_preds_flag, 
                                                save_best_model_per_iteration = save_best_model_per_iteration, 
                                                model_save_path = model_save_path, 
                                                verbose=verbose)

        #returns results as tuple:
        return results_i

    
    
    
    def update_model_weights(self, ts_series_single_area, start_train_year, last_train_set_year, 
                             start_validation_set_year, start_test_set_year, model_name, 
                             end_validation_set_year=None, end_test_set_year=None, n_epochs = 150, n_batch_size = 512, 
                             model_to_update = None, overwrite_params = False, get_preds_flag = True, verbose=0):
        
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
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_list = self.generate_data(ts_series_single_area, 
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
        
        #define optimizer for model:
        if self.clip_norm == True:
            print('#Clipping Norm applied')
            adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0, clipnorm=1.0)
        else:
            adam_opt = optimizers.Adam(lr= self.learning_rate, decay=0.0)
            
        #compile model:
        prediction_model.compile(optimizer=adam_opt, loss='mse', metrics=['mae']) 
                    
        #updating --> valid data not used for fitting:
        history = prediction_model.fit(X_train, y_train, epochs=n_epochs, batch_size = n_batch_size, verbose=1, 
                             shuffle = self.shuffle_flag) 
        
        #assign/"store" model, history & model_name:
        self.prediction_model = prediction_model 
        self.model_name = model_name
        self.training_history = history

        
        
        # 3) get predictions with updated model:
        if get_preds_flag == True:
            #set some parameters to call get_preds_non_state() function properly:
            #necessary to set list to empty list since it is not used by simple MLP
            target_featrs_list = []

            '''#Note: necessary to set "additional_features_flag" to "False" --> the MLP model can incorporate seasonal lags directly! 
                    -> not necessary to create an extra vector, therefore indicate "get_preds_non_state()" 
                    that no extra vector is available'''
            additional_features_flag = False


            #set flag:
            multivariate_flag = False

            if end_validation_set_year == None:
                end_validation_set_year = start_validation_set_year

            #prediction results validation data:
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, additional_features_flag, target_featrs_list,
                                                                              multivariate_flag, self.n_preds, 
                                                                              start_validation_set_year, 
                                                                              end_validation_set_year, True,
                                                                              scaler_list, self.standardizing_flag, self.scale_range,
                                                                              prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_validation_set_year), verbose)

            #prediction results test data:
            if start_test_set_year == None:
                start_test_set_year = start_validation_set_year
            if end_test_set_year == None:
                end_test_set_year = start_test_set_year

            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, additional_features_flag, target_featrs_list,
                                                                              multivariate_flag, self.n_preds, start_test_set_year, 
                                                                              end_test_set_year, valid_flag, 
                                                                              scaler_list, self.standardizing_flag, self.scale_range, 
                                                                              prediction_model, ts_series_single_area, 
                                                                              'results_{}'.format(start_test_set_year), verbose)


            #create results tuple with only prediction results & model: (history does not contain information about validation data!!)
            results_i = (history, prediction_model, predictions_results, model_name, ts_series_single_area,
                         validation_results, rmse_results_test, rmse_results_valid)
        
        
        else:
            print('## Only training history & model are returned')
            results_i = (history, prediction_model)
        
        #returns results as tuple:
        return results_i
        
  
       
       
  
       


  

