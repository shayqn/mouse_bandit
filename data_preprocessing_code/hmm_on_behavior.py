import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import numpy as np
import pandas as pd
import os

def predictBeliefBySession(record_path, session_name, mouse_id, p=0.9, duration=60,
                           root_dir='/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'):

    record = pd.read_csv(record_path,index_col=0)
    
    record[record['Session ID'] == session_name]
    
    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
               'Right Reward Prob','Left Reward Prob','Reward Given',
              'center_frame','decision_frame']
    
    #root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'
    
    full_name = session_name + '_trials.csv'
    
    path_name = os.path.join(root_dir,full_name)
    
    trial_df = pd.read_csv(path_name,names=columns)
    
    data = trial_df.copy()
    
    '''
    Tuned parameters
    '''
    #duration = 60 # number steps until switch reward probability
    #p = 0.8 # prob of reward if choose the correct side
    q = 1.0-p # prob of reward if choose the incorrect side
        
    '''
    Set up outcome & transition matrices T such that T[i,j] is the probability of transitioning
    from state i to state j. 
    If the true number of trials before switching is 'duration', then set the
    probability of switching to be 1 / duration, and the probability of 
    staying to 1 - 1 / duration
    '''
        
    s = 1 - 1./duration
    T = np.array([[s, 1.0-s],
                  [1.0-s,s]])
    
    #observation array
    '''
    set up array such that O[r,z,a] = Pr(reward=r | state=z,action=a)
    
    eg. when action = L, observation matrix should be:
    O[:,:,1]    = [P(r=0 | z=0,a=0),  P(r=0 | z=1,a=0)
                    P(r=1 | z=0,a=0),  P(r=1 | z=1,a=0)]
                = [1-p, 1-q
                    p,   q]
    '''
    O = np.zeros((2,2,2))
    # let a 'right' choice be represented by '0'
    O[:,:,0] = np.array([[1.0-p, 1.0-q],
                        [p,q]])
    O[:,:,1] = np.array([[1.0-q, 1.0-p],
                        [q,p]])
    
    #TEST: All conditional probability distributions must sum to one
    assert np.allclose(O.sum(0),1), "All conditional probability distributions must sum to one!"
        
    """
    Run model on data and output predicted belief state based on evidence from entire session
    """
        
    #set test data
    data_test = data.copy()
    
    n_trials = data_test.shape[0]
    
    #initialize prediction array
    y_predict = np.zeros(n_trials)
    likeli = []
    master_beliefs = np.zeros(n_trials)
    
    for trial in range(data_test.shape[0]):
        n_plays = trial
        actions = data['Port Poked'].values - 1
            # originally Port 2 = left (Decision 1), port 1 = right...switch to zeros and ones (right = 0)
        rewards = data['Reward Given'].values
        beliefs = np.nan*np.ones((n_plays+1,2))
        beliefs[0] = [0.5,0.5] #initialize both sides with equal probability
        #run the algorithm
    
    
        for play in range(n_plays):
    
            assert np.allclose(beliefs[play].sum(), 1.0), "Beliefs must sum to one!"
    
            #update neg log likelihood
            likeli.append(-1*np.log(beliefs[play,actions[play]]))
    
            #update beliefs for next play
            #step 1: multiply by p(r_t | z_t = k, a_t)
            belief_temp = O[rewards[play],:,actions[play]] * beliefs[play]
    
            #step 2: sum over z_t, weighting by transition matrix
            beliefs[play+1] = T.dot(belief_temp)
    
            #step 3: normalize
            beliefs[play+1] /= beliefs[play+1].sum()
    
        #predict action
        y_predict[trial] = np.where(beliefs[-1] == beliefs[-1].max())[0][0]
        master_beliefs[trial] = beliefs[-1][0]
    return master_beliefs


def predictBeliefFeatureMat(data, n_plays, p=0.9, duration=60):
    
    #data = pd.read_csv(data_path, index_col=0) 
    
    """
    Initialize port and reward identities
    """
    
    port_features = []
    reward_features = []

    #change right port to -1 instead of 0
    for col in data:
        if '_Port' in col:
            port_features.append(col)
        elif '_Reward' in col:
            reward_features.append(col)

    '''
    Tuned parameters
    '''
    #duration = 60
    #p = 0.9 # prob of reward if choose the correct side
    q = 1.0-p # prob of reward if choose the incorrect side
    
    '''
    Set up outcome & transition matrices T such that T[i,j] is the probability of transitioning
    from state i to state j. 
    If the true number of trials before switching is 'duration', then set the
    probability of switching to be 1 / duration, and the probability of 
    staying to 1 - 1 / duration
    '''
    
    s = 1 - 1./duration
    T = np.array([[s, 1.0-s],
                 [1.0-s,s]])

    #observation array
    '''
    set up array such that O[r,z,a] = Pr(reward=r | state=z,action=a)

    eg. when action = L, observation matrix should be:
    O[:,:,1]    = [P(r=0 | z=0,a=0),  P(r=0 | z=1,a=0)
                   P(r=1 | z=0,a=0),  P(r=1 | z=1,a=0)]
                = [1-p, 1-q
                    p,   q]
    '''
    O = np.zeros((2,2,2))
    # let a 'right' choice be represented by '0'
    O[:,:,0] = np.array([[1.0-p, 1.0-q],
                         [p,q]])
    O[:,:,1] = np.array([[1.0-q, 1.0-p],
                         [q,p]])

    #TEST: All conditional probability distributions must sum to one
    assert np.allclose(O.sum(0),1), "All conditional probability distributions must sum to one!"
    
    """
    Run model on data and output predicted belief state based on evidence from past 10 trials
    """
    
    #set test data
    data_test = data.copy()

    n_trials = data_test.shape[0]

    #initialize prediction array
    y_predict = np.zeros(n_trials)
    likeli = []
    master_beliefs = np.zeros(n_trials)

    for trial in range(data_test.shape[0]):
        curr_trial = data_test.iloc[trial]
        actions = curr_trial[port_features].values
        rewards = curr_trial[reward_features].values
        beliefs = np.nan*np.ones((n_plays+1,2))
        beliefs[0] = [0.5,0.5] #initialize both sides with equal probability
        #run the algorithm
        for play in range(n_plays):

            assert np.allclose(beliefs[play].sum(), 1.0), "Beliefs must sum to one!"

            #update neg log likelihood
            likeli.append(-1*np.log(beliefs[play,int(actions[play])]))

            #update beliefs for next play
            #step 1: multiply by p(r_t | z_t = k, a_t)
            belief_temp = O[int(rewards[play]),:,int(actions[play])] * beliefs[play]

            #step 2: sum over z_t, weighting by transition matrix
            beliefs[play+1] = T.dot(belief_temp)

            #step 3: normalize
            beliefs[play+1] /= beliefs[play+1].sum()

        #predict action
        y_predict[trial] = np.where(beliefs[-1] == beliefs[-1].max())[0][0]
        master_beliefs[trial] = beliefs[-1][0]
        
    return master_beliefs