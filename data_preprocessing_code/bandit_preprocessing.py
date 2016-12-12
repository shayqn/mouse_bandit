"""
Created on Wed Nov 16 20:32:48 2016

@author: shayneufeld

This file contains preprocessing functions to create feature dataframes
for our AC209A project modeling the 2-armed bandit task in mice
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

def create_feature_matrix(trials,n_indi,mouse_id,session_id,feature_names):
    '''
    This function creates the feature matrix we will use!
    
    Inputs:
            trials       :  pandas dataframe returned by extractTrials.m
            n_indi       :  number of past trials to be used in individual trial features
            mouse_id     : mouse id
            session_id   : sesion_id
            feature_names: list of column names for the datafram
    Outputs:
            feature_trials: pandas dataframe of the features for each trial
    
    Note:
        - this only considers trials 10 to the end
        - it assumes n_summary > n_indi
        
    '''
    n_trials = trials.shape[0] #number of trials in this session
    
    num_cols = 4*n_indi + 2 + 1 + 2 #2 streak rows, 4 cols for each past trial, 1 for current trial + 2 decision/switch!
    feature_matrix = np.zeros((n_trials-n_indi,num_cols))
    
    for j,i in enumerate(np.arange(n_indi,n_trials)):
        
        #extract the 'n_summary' trials we need to consider. Assume that n_summary > n_indi
        past_trials = trials.iloc[i-n_indi:i]
        
        '''
        Mouse ID
        '''
        # will be added after (since its a string)
        
        '''
        Session ID
        '''
        # will be added after (since its a string)
        
        '''
        PORT STREAK
        
        approach: take the derivative of the 'port poked' variabe. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streakP_vec = np.flipud(np.diff(past_trials['Port Poked'].values)) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streakP_len = np.nonzero(streakP_vec)[0]
        
        if len(streakP_len) == 0: #have to deal with case where streak is all 10 previous trials!
            feature_matrix[j,0] = 10
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streakP_len = streakP_len[0]
            feature_matrix[j,0] = (streakP_len+1)
        
        
        '''
        REWARD STREAK
        
        approach: take the derivative of the reward boolean. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streakR_vec = np.flipud(np.diff(past_trials['Reward Given'].values)) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streakR_len = np.nonzero(streakR_vec)[0]
        
        if len(streakR_len) == 0: #have to deal with case where streak is all 10 previous trials!
            streakR_len = 10
            if np.sum(past_trials['Reward Given'].values > 0):
                streakR_sign = 1
            else:
                streakR_sign = -1
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streakR_len = streakR_len[0]
            streakR_sign = streakR_vec[streakR_len]
        
            feature_matrix[j,1] = (streakR_len+1)*streakR_sign
            
        '''
        INDIVIDUAL TRIALS
        '''
        k = 2
        for icol,itrial in enumerate(np.arange(n_indi,0,-1)):
            
            past_trial = past_trials.iloc[-itrial,:]
            
            #which port
            if past_trial['Port Poked'] == 1:
                feature_matrix[j,k] = 0
            elif past_trial['Port Poked'] == 2:
                feature_matrix[j,k] = 1
            else:
                print('Error port not Left or Right')
            k += 1
            
            #reward given:
            feature_matrix[j,k] = past_trial['Reward Given']
            k += 1
            
            #ITI
            feature_matrix[j,k] = past_trial['Since last trial (s)']
            k += 1
            
            #trial time
            feature_matrix[j,k] = past_trial['Trial Duration (s)']
            k += 1
        
        '''
        CURRENT TRIAL
        '''
        current_trial = trials.iloc[i,:]
        feature_matrix[j,k] = current_trial['Since last trial (s)']
        k += 1
    
        '''
        DECISION
        '''
        if current_trial['Port Poked'] == 1:
            feature_matrix[j,k] = 0
        elif current_trial['Port Poked'] == 2:
            feature_matrix[j,k] = 1
        else:
            print('Error decision port not Left or Right')
        
        k += 1
        '''
        SWITCH
        '''
        feature_matrix[j,k] = np.abs((current_trial['Port Poked'] - trials.iloc[i-1]['Port Poked']))
    
    
    d = {'Mouse ID':mouse_id,'Session ID':session_id}
    feature_trials = pd.DataFrame(data=d,index=range(feature_matrix.shape[0]))
    feature_trials = pd.concat([feature_trials,pd.DataFrame(data=feature_matrix,index=None,columns=feature_names)],axis=1)
    
    return feature_trials
    
def OneHotEncode(data):
    
    categorical = (data.dtypes.values != np.dtype('float64'))
    data_1hot = data.apply(encode_categorical)
    encoder = preprocessing.OneHotEncoder(categorical_features=categorical, sparse=False)  # Last value in mask is y
    data_encoded = encoder.fit_transform(data_1hot.values)
    
    return data_encoded
    
def encode_categorical(array):
    if not (array.dtype == np.dtype('float64') or array.dtype == np.dtype('int64')) :
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array