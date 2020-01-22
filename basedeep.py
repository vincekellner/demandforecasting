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





class BaseDeepClass(object):
    
    def __init__(self):
        
        pass
        
        
    def get_params(self):
        
        pass
    

    def load_model(self, model_to_load):
        
        pass
    
    
    def differencing(self, dataset, shift_interval):
        '''
        #function makes timeseries stationary through differencing
        '''
        
        dataset_copy = dataset.copy(deep=True)
        shifted_df = dataset_copy.shift(shift_interval)
        diff_series = dataset_copy - shifted_df
        diff_series.dropna(inplace=True)
        return diff_series



    def invert_differencing(self, history, prediction, start_year_to_invert, end_year_to_invert):
        '''
        # function inverts differenced values 
        '''
        history_copy = history.copy()
        history_shift = history_copy.shift(1)
        history_shift.dropna(inplace=True)

        print('Shape of org. dataset after shift: ', history_shift.loc[start_year_to_invert:end_year_to_invert].shape)

        history_shift_year =  history_shift.loc[start_year_to_invert:end_year_to_invert]

        return prediction + history_shift_year

    
    
    
       
    def data_scaler(self, train_data, valid_data, test_data, scale_range, standardizing_flag=False, verbose=0):
        
        '''
        #function scales input data either through standardization or given scale_range
        #it is recommended to scale input data in the range of the activation function used by the network. 
        # datasets are scaled based on fit of train_set; 
        Note: input set has to be a reshaped numpy_array!
        
        '''

        if standardizing_flag == False: 
            if verbose == 1:
                print('MinMax-Scaling used...')
            #use MinMax-Scaling based on scale_range given...
            scaler = MinMaxScaler(feature_range=scale_range) #feature_range=(-1, 1)
            scaler = scaler.fit(train_data)
            #scale train_set:
            train_scaled = scaler.transform(train_data)
            train_scaled_df = pd.DataFrame(train_scaled)
            #scale valididation data based on scaler fitted on training data:
            valid_scaled = scaler.transform(valid_data)
            valid_scaled_df = pd.DataFrame(valid_scaled)
            #scale test_set based on scaler fitted on training data:
            test_scaled = scaler.transform(test_data)
            test_scaled_df = pd.DataFrame(test_scaled)

        else:  
            if verbose == 1:
                print('Standardizing used...')

            scaler = StandardScaler()
            scaler = scaler.fit(train_data)
            #scale train_set:
            train_scaled = scaler.transform(train_data)
            train_scaled_df = pd.DataFrame(train_scaled)
            #scale valididation data based on scaler fitted on training data:
            valid_scaled = scaler.transform(valid_data)
            valid_scaled_df = pd.DataFrame(valid_scaled)
            #scale test_set based on scaler fitted on training data:
            test_scaled = scaler.transform(test_data)
            test_scaled_df = pd.DataFrame(test_scaled)


        #return data as df, since "supervised" function requires df or series as input
        return scaler, train_scaled_df, valid_scaled_df, test_scaled_df


    
    
    def invert_data_scaler(self, scaler, predictions_data, standardizing_flag, scale_range, n_preds):
        
        '''
        
        #function to invert scaling:
        '''

        if standardizing_flag == False: 
            #create new scaler: -> necessary since old scaler might be fitted on lagged values 
            #-> expects more dimensions as the predictions have! 
            new_scaler = MinMaxScaler(feature_range=scale_range)
            #copy attributes of old scaler to new one: but only for the first columns for which we have real predictions
            new_scaler.min_, new_scaler.scale_ = scaler.min_[:n_preds], scaler.scale_[:n_preds]

            preds = new_scaler.inverse_transform(predictions_data)

        else:
            new_scaler = StandardScaler()
            #copy attributes of old scaler to new one: but only for the first columns for which we have real predictions
            new_scaler.mean_, new_scaler.scale_ = scaler.mean_[:n_preds], scaler.scale_[:n_preds]

            preds = new_scaler.inverse_transform(predictions_data)


        return preds



    
    
    
    def create_supervised_data_single_ts(self, single_ts, n_timesteps, seasonal_lags_flag, seasonal_lag_set):
        
        '''
        function to create lags of data and seasonal lags, data is rearranged in a "supervised" 
        format such it can be directly used for modeling
        number of lags = n_timesteps
        seasonal lags = lags based on target for multiple weeks --> this way weekly seasonality can be "captured"
        
        '''
        
        #copy dfs to ensure no changes are made on original dataset:
        sequence_copy = single_ts.copy(deep=True)
        #create dfs to easily append new columns and access columns:
        sequence_copy = pd.DataFrame(sequence_copy)
        sequence_df = pd.DataFrame(sequence_copy)

        #Note: range starts reversed to make sure the order of the sequence is correct:
        for i in range(n_timesteps,0,-1): 
            sequence_df['lag_{}'.format(i)] = sequence_copy.iloc[:,0].shift(i)

        if seasonal_lags_flag == True:
            #add seasonal lagged values of first column to df:       
            for u in range(len(seasonal_lag_set)):
                sequence_df['season_lag{}'.format(u+1)] = sequence_copy.iloc[:,0].shift(seasonal_lag_set[u])

        #drop rows with NaNs -> if no lagged features are available, we drop the whole row
        sequence_df.dropna(inplace=True) 

        return sequence_df  
    
    
    
    
    
    
    def create_one_hot_encoded_areas(self, multivar_series):
        
        '''
        #function returns one-hot-encoded areas:
        
        '''
        
        #Note: only works correctly if no additional features are already appended to multivar_series
        multivar_series_copy = multivar_series.copy()

        area_labels = list(multivar_series_copy.columns)

        all_df_encoded = pd.DataFrame()

        #append ts of each area to create one big df:
        for i in range(len(area_labels)):
            #select single area:
            single_ts = pd.DataFrame(multivar_series_copy.iloc[:,i])
            #prepare df:
            #drop 'datetime' to append dfs on axis=0
            single_ts = single_ts.reset_index().drop('date', axis=1)
            #rename first column: (necessary to append all dfs on axis=0)
            single_ts = single_ts.rename(columns={'{}'.format(area_labels[i]):'area_values'})
            #add additional columns for label:
            single_ts['area_label'] = area_labels[i]
            single_ts['area'] = area_labels[i]

            #append single ts:
            all_df_encoded = pd.concat([all_df_encoded,single_ts], axis=0)

        #create dummies:
        all_dummies_df = pd.get_dummies(all_df_encoded,prefix=['encod'],columns=['area_label'])

        #restore original ts for each area but with one-hot-encoded area_labels:
        #store results in dict:
        dict_areas = {}
        for i in range(len(area_labels)):
            #slice individual area:
            single_ts_restored = all_dummies_df[all_dummies_df['area']==area_labels[i]]
            #restore index:
            single_ts_restored.index = multivar_series_copy.index
            #rename columns:
            single_ts_restored = single_ts_restored.rename(columns={'area_values':'{}'.format(area_labels[i])})
            #drop label column:
            single_ts_restored.drop('area',axis=1,inplace=True)

            #append single_ts to dict:
            new_key = 'area' + area_labels[i]
            dict_areas[new_key] = single_ts_restored

        return dict_areas


    

    
    
    def get_day_of_week_features_one_hot_encoded(self, ts_series):
        
        '''
        #function returns one-hot-encoded weekdays for a given ts:
        '''
        
        sequence_copy = ts_series.copy()
        if len(sequence_copy.shape) > 1:
            #slice sequence_copy to only keep first column (needed if ts_series = multivariate-ts):
            sequence_copy = sequence_copy.iloc[:,0]
        #create df to easily add new column:
        sequence_copy = pd.DataFrame(sequence_copy)

        #create dayofweek values:
        help_series = pd.Series(list(ts_series.index.dayofweek), index=ts_series.index)
        sequence_copy['dayofweek'] = help_series

        #create one-hot-encoding of dayofweek values:
        dummies = pd.get_dummies(sequence_copy,prefix=['dayofweek'],columns=['dayofweek'])

        #return only dummies:
        dummies = dummies.iloc[:,1:]

        return dummies


    
      
    
    def get_hour_of_day_features_sin_cos_encoded(self, ts_series):
        
        '''
        ##function returns sin-cos encoded hours for a given ts:
        '''
        
        sequence_copy = ts_series.copy()
        if len(sequence_copy.shape) > 1:
            #slice sequence_copy to only keep first column (needed if ts_series = multivariate-ts):
            sequence_copy = sequence_copy.iloc[:,0]
        #create df to easily add new column:
        sequence_copy = pd.DataFrame(sequence_copy)

        #create dayofweek values:
        help_series = pd.Series(list(ts_series.index.hour), index=ts_series.index)
        sequence_copy['hourofday'] = help_series

        #map hourofday onto unit circle:
        sequence_copy['sin_hour'] = np.sin(2*np.pi*sequence_copy['hourofday']/24)
        sequence_copy['cos_hour'] = np.cos(2*np.pi*sequence_copy['hourofday']/24)

        #drop column which is not needed anymore:
        sequence_copy.drop('hourofday', axis=1, inplace=True)

        #return only encoded features of df:
        encoded_feats = sequence_copy.iloc[:,1:]


        return encoded_feats


    
    
    def get_month_of_year_features_sin_cos_encoded(self, ts_series):
        
        '''
        #function returns sin-cos encoded month for a given ts:
        '''
        
        sequence_copy = ts_series.copy()
        if len(sequence_copy.shape) > 1:
            #slice sequence_copy to only keep first column (needed if ts_series = multivariate-ts):
            sequence_copy = sequence_copy.iloc[:,0]
        #create df to easily add new column:
        sequence_copy = pd.DataFrame(sequence_copy)

        #create dayofweek values:
        help_series = pd.Series(list(ts_series.index.month), index=ts_series.index)
        sequence_copy['monthofyear'] = help_series

        #map hourofday onto unit circle:
        sequence_copy['sin_month'] = np.sin(2*np.pi*sequence_copy['monthofyear']/12)
        sequence_copy['cos_month'] = np.cos(2*np.pi*sequence_copy['monthofyear']/12)

        #drop column which is not needed anymore:
        sequence_copy.drop('monthofyear', axis=1, inplace=True)

        #return only encoded features of df:
        encoded_feats = sequence_copy.iloc[:,1:]

        return encoded_feats


    
    

    '''
    def get_encoded_features_for_prev_lags(self, df_with_encodings, n_timesteps):
        

        #returns encoded features (weekofday, hourofday...) for each lag (n_timesteps) of target 
        #-> these features can be concatenated on axis = 2 for LSTM..

        #print('# create additional lagged features')
        
        #Note: input df is assumend to already contain encodings of weekofday, hourofday etc. in seaparate columns for each date
        #NOTE: it is assumed that actuals are stored in first column of df and encodings in remaining columns..
        all_encodings_of_lag_windows = list()

        #for each window of "n_timesteps" slice encodings:
        for i in range(len(df_with_encodings)):
            #define index of window:
            end_idx = i + n_timesteps
            # check if we already processed whole series
            if end_idx > len(df_with_encodings)-1:
                break
            # get encodings of each lag (for each previous date: if n_timesteps = 24 -> get encodings for each timestep)
            encodings_of_lag_window = df_with_encodings.iloc[i:end_idx,:].values #slice encodings
            all_encodings_of_lag_windows.append(encodings_of_lag_window)
            
        #print('## lag encodings list[0]')
        #print(all_encodings_of_lag_windows[0])

        #Note: important to return "np.array()" -> this way we already have the correct shape
        return np.array(all_encodings_of_lag_windows) 

    '''
    
    
    def get_encoded_feature_matrix_for_prev_lags(self, df_with_encodings, n_timesteps, verbose=0):

        '''
        #returns encoded features (weekofday, hourofday...) for each lag (n_timesteps) of target 
        -> these features can be concatenated on axis = 2 for LSTM..
        the features are returned in a dict since a matrix is created for each feature which is later reshaped into a tensor (with function: 'get_encoding_tensor()')
        '''
        #print('# create additional lagged features')

        #Note: input df is assumend to already contain encodings of weekofday, hourofday etc. in seaparate columns for each date
        #NOTE: it is assumed that actuals are stored in first column of df and encodings in remaining columns..

        label_list = list(df_with_encodings.columns)

        prev_encodings_dict = {}

        #for each encoding create sequence of size = n_timesteps
        for i in range(df_with_encodings.shape[1]):
            
            if verbose > 1:
                print('create feature matrix of encodings for: ', label_list[i])

            #copy dfs to ensure no changes are made on original dataset:
            df_copy_shifter = pd.DataFrame(df_with_encodings.iloc[:,i].copy(deep=True))

            #create dfs to easily append new columns and access columns:
            df_copy_df = pd.DataFrame(df_with_encodings.iloc[:,i].copy(deep=True))

            #Note: range starts reversed to make sure the order of the sequence is correct:
            for u in range(n_timesteps,0,-1): 
                df_copy_df['lag_{}_{}'.format(label_list[i],u)] = df_copy_shifter.iloc[:,0].shift(u)


            #drop rows with NaNs -> if no lagged features are available, we drop the whole row
            df_copy_df.dropna(inplace=True) 
            #drop first column of df:
            df_copy_df.drop(df_copy_df.columns[0], axis=1, inplace=True)

            #store lagged features in dict:
            prev_encodings_dict[label_list[i]] = df_copy_df

        return prev_encodings_dict




    def get_encoding_tensor(self, prev_encodings_dict, start_train_year, end_train_year, start_valid_year=None, end_valid_year=None,
                           start_test_year=None, end_test_year=None, verbose=0):


        key_list = list(prev_encodings_dict.keys())

        #adjust dates:
        if end_valid_year == None:
            end_valid_year = start_valid_year
        if end_test_year == None:
            end_test_year = start_test_year

        year_list = [(start_train_year,end_train_year), (start_valid_year,end_valid_year),(start_test_year,end_test_year)]

        feat_tensors_list = []

        for u in range(len(year_list)):
            
            if verbose > 1:
                print('> create tensor for years {}:{}'.format(year_list[u][0],year_list[u][1]))

            for i in range(len(key_list)):

                if i < 1:
                    feat_tensor = prev_encodings_dict[key_list[i]].loc[year_list[u][0]:year_list[u][1]].values
                    feat_tensor = feat_tensor.reshape((feat_tensor.shape[0],feat_tensor.shape[1],1))
                else:
                    help_tensor = prev_encodings_dict[key_list[i]].loc[year_list[u][0]:year_list[u][1]].values
                    help_tensor = help_tensor.reshape((help_tensor.shape[0],help_tensor.shape[1],1))
                    feat_tensor = np.concatenate((feat_tensor,help_tensor), axis=2)

            feat_tensors_list.append(feat_tensor)


        return feat_tensors_list


    

    def generate_data(self):
        
        pass
    
    
    
            
    def create_model(self):
        
        pass
    
    
    def create_full_pred_model(self):
        pass
      
    
    
    
    def get_preds_non_state(self, X_test, external_feat_flag, external_feat_list, multivariate_flag, n_preds, 
                            start_year_to_invert, end_year_to_invert, valid_flag, scaler_list, standardizing_flag, scale_range, model,
                            original_complete_dataset, model_name, verbose=0):
        
        '''
        function gets predictions for non_stateful models (assuming scaling + differencing was applied on dataset):
        function distinguishes between multivariate predictions and single time series predictions 
        --> therefore "multivariate_flag" needed
        
        returns predictions as df + RMSE results in a list (RMSE for each area)
        
        '''
    
    
        #check if external features are used:
        if external_feat_flag == True:
            #check if validation data or test data need to be accessed:
            if valid_flag == True:
                external_features_used = external_feat_list[1]
            else:
                external_features_used = external_feat_list[2]

            #get predictions based on sequence input & external input:
            yhat_s = model.predict([X_test, external_features_used], verbose=1)

        else:
            #get predictions based on sequence input:
            yhat_s = model.predict(X_test, verbose=1)
            #print('First 10 predictions non-scaled')
            #print(yhat_s[0:10])

        #rescale predictions:
        if multivariate_flag == False:
            yhat_rescaled_all = self.invert_data_scaler(scaler_list[0], yhat_s, standardizing_flag, scale_range, n_preds)

        else:
            yhat_rescaled_all_list = []
            for i in range(len(scaler_list)):
                #slice predictions for each area and rescale predictions: (Note: slicing columns of numpy array returns a list! 
                #-> reshaping necessary afterwards)
                yhat_area_i = yhat_s[:,i]
                #reshape unscaled predictions for scaler:
                yhat_area_i = yhat_area_i.reshape(len(yhat_area_i),1)
                #apply scaler:
                yhat_rescaled_area_i = self.invert_data_scaler(scaler_list[i], yhat_area_i, standardizing_flag, scale_range, 1) 
                #Note: for multivariate case n_preds is set to "1" since we only take first column of scaler for each area
                yhat_rescaled_all_list.append(yhat_rescaled_area_i)

            #restore numpy_array based on yhat_rescaled_all_list: (this way we have all rescaled predictions for each area 
            # in one big numpy array)
            yhat_rescaled_all = yhat_rescaled_all_list[0]
            for i in range(1,len(yhat_rescaled_all_list)):
                yhat_rescaled_all = np.concatenate((yhat_rescaled_all,yhat_rescaled_all_list[i]),axis=1)

        if verbose == 1:
            print('First 2 scaled predictions')
            print(yhat_rescaled_all[0:2])
            print('Shape of predictions:', yhat_rescaled_all.shape)

        #compare predictions with actuals:    
        if multivariate_flag == True:
            if verbose == 1:
                print('Invert Differencing of multivariate predictions...')    
            #invert differencing:  (adding value of previous timestep)
            predictions_all = self.invert_differencing(original_complete_dataset, yhat_rescaled_all, 
                                                       start_year_to_invert, end_year_to_invert)
            
            if verbose == 1:
                print('predictions preview:')
                print(predictions_all.head())

            #get rmse for each timeseries
            rmse_per_ts = []
            for u in range(n_preds):
                rmse_single_ts = np.sqrt(
                                mean_squared_error(original_complete_dataset.loc[start_year_to_invert:end_year_to_invert].iloc[:,u],
                                                            predictions_all.iloc[:,u]))
                rmse_per_ts.append(rmse_single_ts)
                if verbose == 1:
                    print('RMSE per TS {} for model: {}: {}'.format(u, model_name, rmse_per_ts[u]))

            #get average of all rmses
            total_rmse = np.mean(rmse_per_ts)
            if verbose == 1:
                print('Avg.RMSE for multivariate model: {}: {}'.format(model_name, total_rmse))

        else:
            #invert differencing:  (adding value of previous timestep)
            if verbose == 1:
                print('Invert Differencing of predictions...')        
            predictions_all = self.invert_differencing(original_complete_dataset, yhat_rescaled_all[:,0], 
                                                       start_year_to_invert, end_year_to_invert)
            
            if verbose == 1:
                print('predictions preview:')
                print(predictions_all.head())

            #get rmse:
            rmse = np.sqrt(mean_squared_error(original_complete_dataset[start_year_to_invert:end_year_to_invert], predictions_all))
            if verbose == 1:
                print('RMSE for model: {}: {}'.format(model_name, rmse))


        #return RMSE results:
        rmse_results = []
        if multivariate_flag == True:
            rmse_results.append(total_rmse)
            rmse_results.append(rmse_per_ts)

        else:
            rmse_results.append(rmse)


        return predictions_all, rmse_results

    
    
    
    
        
    
    
    
    

