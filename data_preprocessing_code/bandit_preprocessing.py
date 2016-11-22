"""
Created on Wed Nov 16 20:32:48 2016

@author: shayneufeld

This file contains preprocessing functions to create feature dataframes
for our AC209A project modeling the 2-armed bandit task in mice
"""
import numpy as np
import pandas as pd

def create_feature_matrix(df,n_summary,n_indi,feature_names):
    '''
    This function creates the feature matrix we will use!
    
    Inputs:
            df       :  pandas dataframe returned by extractTrials.m
            n_summary:  number of past trials to be used in summary features
            n_indi   :  number of past trials to be used in individual trial features
            feature_names: list of column names for the datafram
    Outputs:
            feature_df: pandas dataframe of the features for each trial
    
    Note:
        - this only considers trials 10 to the end
        - it assumes n_summary > n_indi
        
    '''
    n_trials = df.shape[0] #number of trials in this session
    
    num_cols = 5 + 4*n_indi + 2 + 1 #5 summary rows, 4 cols for each past trial, 
    #2 for current trial + 1 decision!
    feature_matrix = np.zeros((n_trials-n_summary,num_cols))
    
    for j,i in enumerate(np.arange(n_summary,n_trials)):
    
        #extract the 'n_summary' trials we need to consider. Assume that n_summary > n_indi
        past_trials = df.iloc[i-n_summary:i]
        
        #calculate summary statistics
        
        #left and right ports
        feature_matrix[j,0] = np.sum(past_trials['Port Poked'].values == 2)
        feature_matrix[j,1] = np.sum(past_trials['Port Poked'].values == 1)
        
        #left and right rewards
        feature_matrix[j,2] = np.sum(past_trials.loc[past_trials['Port Poked'].values == 2,'Reward Given'])
        feature_matrix[j,3] = np.sum(past_trials.loc[past_trials['Port Poked'].values == 1,'Reward Given'])
        
        #streak
        '''
        approach: take the derivative of the reward boolean. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streak_vec = np.flipud(np.diff(past_trials['Reward Given'].values)) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streak_len = np.nonzero(streak_vec)[0]
        
        if len(streak_len) == 0: #have to deal with case where streak is all 10 previous trials!
            streak_len = 10
            if np.sum(past_trials['Reward Given'].values > 0):
                streak_sign = 1
            else:
                streak_sign = -1
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streak_len = streak_len[0]
            streak_sign = streak_vec[streak_len]
        
            feature_matrix[j,4] = (streak_len+1)*streak_sign
        
        '''
        INDIVIDUAL TRIALS
        '''
        k = 0
        for icol,itrial in enumerate(np.arange(n_indi,0,-1)):
            
            past_trial = past_trials.iloc[-itrial,:]
            
            #which port
            if past_trial['Port Poked'] == 1:
                feature_matrix[j,5+k] = 0
            elif past_trial['Port Poked'] == 2:
                feature_matrix[j,5+k] = 1
            else:
                print('Error port not Left or Right')
            k += 1
            
            #reward given:
            feature_matrix[j,5+k] = past_trial['Reward Given']
            k += 1
            
            #ITI
            feature_matrix[j,5+k] = past_trial['Since last trial (s)']
            k += 1
            
            #trial time
            feature_matrix[j,5+k] = past_trial['Trial Duration (s)']
            k += 1
        
        '''
        CURRENT TRIAL
        '''
        current_trial = df.iloc[i,:]
        feature_matrix[j,5+k] = current_trial['Since last trial (s)']
        k += 1
        feature_matrix[j,5+k] = current_trial['Trial Duration (s)']
        k += 1
        '''
        DECISION
        '''
        if current_trial['Port Poked'] == 1:
            feature_matrix[j,5+k] = 0
        elif current_trial['Port Poked'] == 2:
            feature_matrix[j,5+k] = 1
        else:
            print('Error decision port not Left or Right')
    
    feature_df = pd.DataFrame(data=feature_matrix,index=None,columns=feature_names)
    
    return feature_df