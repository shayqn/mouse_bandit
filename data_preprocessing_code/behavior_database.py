#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:56:16 2017

@author: shayneufeld
"""

import numpy as np
import pandas as pd
import os

def add_multi_sessions(root_dir,record_df):

    '''
    This function was written for the 'SuperMice' round of mice (Fall 2016)
    Specifically, where the data is organied by mouse, and in each mouse's
    folder there are folders for each day.
    '''
    output_df = record_df.copy()
    
    names = [
        'Session ID',
        'Mouse ID',
        'Date',
        'Phase',
        'Left Reward Prob',
        'Right Reward Prob',
        'Block Range Min',
        'Block Range Max',
        'No. Trials',
        'No. Blocks',
        'No. Rewards',
        'p(high Port)',
        'Decision Window Duration',
        'Min Inter-trial-interval',
        'Left Solenoid Duration',
        'Right Solenoid Duration'
    ]

    for curr_dir in os.listdir(root_dir):
    
        '''
        load in trial data
        '''
        columns = ['Elapsed Time (s)','Since last trial (s)',
        'Trial Duration (s)','Port Poked','Right Reward Prob',
        'Left Reward Prob','Reward Given']
        
        try:  
            for file in os.listdir(os.path.join(root_dir,curr_dir)):
                if not file[0] == '.':
                    file_name = os.path.join(root_dir,curr_dir,file)
                    
                    if 'trials.csv' in file:
                        trials = pd.read_csv(file_name,names=columns)
                        session_id = file[:file.index('_',9)]
                        
                    elif 'parameters.csv' in file:
                        params = pd.read_csv(file_name)

                else:
                    raise ValueError('In a hidden folder')
                        
            
            high_p_port = np.zeros(trials.shape[0])
            
            for row in trials.iterrows():
                i = row[0]
                current_trial = row[1]
                
                if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
                    high_p_port[i] = 1
                elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
                    high_p_port[i] = 1
                    
            if params['centerPokeTrigger'].values == 0:
                phase = 0
            elif (params['leftRewardProb'].values == params['rightRewardProb'].values):
                phase = 1
            else:
                phase = 2
                
            record = {
                'Session ID': session_id,
                'Mouse ID': session_id[9:],
                'Date': session_id[:8],
                'Phase': phase,
                'Left Reward Prob': params['leftRewardProb'].values,
                'Right Reward Prob': params['rightRewardProb'].values,
                'Block Range Min': params['blockRangeMin'],
                'Block Range Max': params['blockRangeMax'],
                'No. Trials': trials.shape[0],
                'No. Blocks': np.sum(np.diff(trials['Right Reward Prob'].values) != 0),
                'No. Rewards': np.sum(trials['Reward Given']),
                'p(high Port)': np.round(high_p_port.mean(),decimals=2),
                'Decision Window Duration': params['centerPokeRewardWindow'],
                'Min Inter-trial-interval': params['minInterTrialInterval'],
                'Left Solenoid Duration': params['rewardDurationLeft'],
                'Right Solenoid Duration': params['rewardDurationRight']  
                     }
                     
            output_df = output_df.append(pd.DataFrame(data=record,columns=names),ignore_index=True)
        
        except:
            pass
    
    output_df = output_df.drop_duplicates()
    
    return output_df

    
def add_session(root_dir,record_df):
    '''
    This function is for a single session. The root dir here should point
    directly to the folder where trials.csv and parameters.csv are located.
    
    '''
    output_df = record_df.copy()
    
    names = [
        'Session ID',
        'Mouse ID',
        'Date',
        'Phase',
        'Left Reward Prob',
        'Right Reward Prob',
        'Block Range Min',
        'Block Range Max',
        'No. Trials',
        'No. Blocks',
        'No. Rewards',
        'p(high Port)',
        'Decision Window Duration',
        'Min Inter-trial-interval',
        'Left Solenoid Duration',
        'Right Solenoid Duration'
    ]

    
    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)',
    'Trial Duration (s)','Port Poked','Right Reward Prob',
    'Left Reward Prob','Reward Given']
    
    try:  
        for file in os.listdir(root_dir):
            if not file[0] == '.':
                file_name = os.path.join(root_dir,file)
                
                if 'trials.csv' in file:
                    trials = pd.read_csv(file_name,names=columns)
                    session_id = file[:file.index('_',9)]
                    
                elif 'parameters.csv' in file:
                    params = pd.read_csv(file_name)

            else:
                raise ValueError('In a hidden folder')
                    
        
        # calculate p(high port)
        high_p_port = np.zeros(trials.shape[0])
        
        for row in trials.iterrows():
            i = row[0]
            current_trial = row[1]
            
            if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
                high_p_port[i] = 1
            elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
                high_p_port[i] = 1
       
        # determine what stage in training we are in        
        if params['centerPokeTrigger'].values == 0:
            phase = 0
        elif (params['leftRewardProb'].values == params['rightRewardProb'].values):
            phase = 1
        else:
            phase = 2
           
        # convert date to datetime object
        date = session_id[:8]
        date_str =  np.str(date)
        if (len(date_str) == 7):
            date_str = '0' + date_str
        datetime = pd.to_datetime(date_str,format='%m%d%Y')
            
        # create a dictionary with all information
        record = {
            'Session ID': session_id,
            'Mouse ID': session_id[9:],
            'Date': datetime,
            'Phase': phase,
            'Left Reward Prob': params['leftRewardProb'].values,
            'Right Reward Prob': params['rightRewardProb'].values,
            'Block Range Min': params['blockRangeMin'],
            'Block Range Max': params['blockRangeMax'],
            'No. Trials': trials.shape[0],
            'No. Blocks': np.sum(np.diff(trials['Right Reward Prob'].values) != 0),
            'No. Rewards': np.sum(trials['Reward Given']),
            'p(high Port)': np.round(high_p_port.mean(),decimals=2),
            'Decision Window Duration': params['centerPokeRewardWindow'],
            'Min Inter-trial-interval': params['minInterTrialInterval'],
            'Left Solenoid Duration': params['rewardDurationLeft'],
            'Right Solenoid Duration': params['rewardDurationRight']  
                 }
        
        #create DataFrame         
        output_df = output_df.append(pd.DataFrame(data=record,columns=names),ignore_index=True)
    
    except:
        pass
    
    
    output_df = output_df.drop_duplicates()
    
    return output_df
    

def add_old_session(root_dir,record_df):
    '''
    This function is specifically for data collected before we implemeneted 
    the GUI on July 22th, 2016. Before that point, the parameter files were
    csv (not in the same format as above). So we need to deal with it differently.
    
    '''
    output_df = record_df.copy()
    
    names = [
        'Session ID',
        'Mouse ID',
        'Date',
        'Phase',
        'Left Reward Prob',
        'Right Reward Prob',
        'Block Range Min',
        'Block Range Max',
        'No. Trials',
        'No. Blocks',
        'No. Rewards',
        'p(high Port)',
        'Decision Window Duration',
        'Min Inter-trial-interval',
        'Left Solenoid Duration',
        'Right Solenoid Duration'
    ]

    
    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)',
    'Trial Duration (s)','Port Poked','Right Reward Prob',
    'Left Reward Prob','Reward Given']
    
    try:  
        for file in os.listdir(root_dir):
            if not file[0] == '.':
                file_name = os.path.join(root_dir,file)
                
                if 'trials.csv' in file:
                    trials = pd.read_csv(file_name,names=columns)
                    session_id = file[:file.index('_',9)]
                    
                elif 'Parameters' in file:
                    params = pd.read_csv(file_name)
                    param_vals = params['pVal'].values

            else:
                raise ValueError('In a hidden folder')
                    
        
        high_p_port = np.zeros(trials.shape[0])
        
        for row in trials.iterrows():
            i = row[0]
            current_trial = row[1]
            
            if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
                high_p_port[i] = 1
            elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
                high_p_port[i] = 1
                
        if param_vals[0] == 0:
            phase = 0
        elif param_vals[3] == param_vals[4]:
            phase = 1
        else:
            phase = 2
            
        if param_vals.shape[0] == 9:
            block_range_min = param_vals[8]
            block_range_max = param_vals[8]
        else:
            block_range_min = param_vals[8]
            block_range_max = param_vals[9]
                    
        record = {
            'Session ID': session_id,
            'Mouse ID': session_id[9:],
            'Date': session_id[:8],
            'Phase': phase,
            'Left Reward Prob': param_vals[3],
            'Right Reward Prob': param_vals[4],
            'Block Range Min': block_range_min,
            'Block Range Max': block_range_max,
            'No. Trials': trials.shape[0],
            'No. Blocks': np.sum(np.diff(trials['Right Reward Prob'].values) != 0),
            'No. Rewards': np.sum(trials['Reward Given']),
            'p(high Port)': np.round(high_p_port.mean(),decimals=2),
            'Decision Window Duration': param_vals[1],
            'Min Inter-trial-interval': param_vals[7],
            'Left Solenoid Duration': param_vals[6],
            'Right Solenoid Duration': param_vals[5], 
                 }
                 
        output_df = output_df.append(pd.DataFrame(data=record,columns=names,index=np.arange(16)),ignore_index=True)
    
    except:
        pass
    
    
    output_df = output_df.drop_duplicates()
    
    return output_df