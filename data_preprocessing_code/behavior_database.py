#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:56:16 2017

@author: shayneufeld
"""

import numpy as np
import pandas as pd
import os

def add_sessions(root_dir,record_df):

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
        
    return output_df
