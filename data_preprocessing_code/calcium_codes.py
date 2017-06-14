import numpy as np
import scipy.io as scio
import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import pandas as pd
import bandit_preprocessing as bp
import sys
import os


def detectEvents(ca_data_path):

    """
    load in the neuron_master.mat structure as a numpy array
    """

    ca_data = scio.loadmat(ca_data_path, squeeze_me = True, struct_as_record = False)
    neuron = ca_data['neuron_results'] 

    # set model traces to its own variable
    traces = neuron.C
    neuron.Coor = neuron.centers

    n_neurons = traces.shape[0] #number of neurons

    nan_neurons = []
    for iNeuron in range(0,n_neurons):
        if any(np.isnan(neuron.C[iNeuron,:])) == True:
            nan_neurons.append(iNeuron)
    more_nan_neurons = np.where(np.isnan(neuron.Coor))
    nan_neurons.extend(x for x in more_nan_neurons[0] if x not in nan_neurons)
    good_neurons = [x for x in range(0, n_neurons) if x not in nan_neurons]

    # remove neurons with NaNs from relevant variables
    neuron.C = neuron.C[good_neurons, :]
    neuron.Coor = neuron.Coor[good_neurons,:]
    neuron.C_raw = neuron.C_raw[good_neurons,:]

    # refined count for neurons after cleansing NaNs
    nNeurons = neuron.C.shape[0]
    
    """
    set up system to detect events:
        first detect any events above a threshold
        then make sure the events are a least a minimum duration?
        only mark events at the beginning
    """
    
    #Create Gaussian filter and apply to raw trace
    sigma = 1.5;
    sz = 5;  

    x = np.linspace(-sz / 2, sz / 2, sz);
    gaussFilter = np.exp(-x**2 / (2*sigma**2));
    gaussFilter = gaussFilter / np.sum(gaussFilter);

    smoothed = np.zeros((nNeurons, neuron.C_raw.shape[1]+sz-1));

    for i in range(0, nNeurons):
        smoothed[i,:] = np.convolve(neuron.C_raw[i,:], gaussFilter);
        
    """
    Z-score neurons and set threshold for events. Shift by 1 and subtract to produce 0s and 1s.
    """
    
    z_neuron = np.zeros((nNeurons, neuron.C_raw.shape[1]))
    for i in range(0,nNeurons):
        z_neuron[i,:] = (neuron.C_raw[i,:] - np.mean(neuron.C_raw[i,:], axis=0)) / np.std(neuron.C_raw[i,:], axis=0)
    thresh = 5.
    thresh_neuron = z_neuron > thresh

    thresh_shift = np.insert(thresh_neuron, 0, 0 , axis=1)
    thresh_shift = thresh_shift[:,0:thresh_shift.shape[1]-1]
    
    """
    Remove timepoints of decay so events only mark onset times
    """
    
    events_on_off = thresh_neuron - thresh_shift
    events = events_on_off 

    for iNeuron in range(0,nNeurons):
        indices = np.nonzero(events[iNeuron,:])
        for ind in range(0,np.size(indices)):
            if smoothed[iNeuron,indices[0][ind]] - smoothed[iNeuron,(indices[0][ind]-1)]<0:
                events[iNeuron,indices[0][ind]] = 0
                indices[0][ind] = 0
            if ind>0:
                if indices[0][ind]-4 <= indices[0][ind-1]:
                    events[iNeuron, indices[0][ind]] = 0
                    #indices[0][ind] = 0
            
    #median absolute deviation
    #from statsmodels import robust
    #med_noise = robust.mad(neuron.C_raw, axis = 1)

    return(events)



def alignFrames(record_path, ca_data_path, session_name, mouse_id, cond1, cond2, extension=30):

    record = pd.read_csv(record_path,index_col=0)
    ca_data = scio.loadmat(ca_data_path,squeeze_me = True, struct_as_record = False)
    neuron = ca_data['neuron_results'] 
    
    record[record['Session ID'] == session_name]
    
    """
    Extract data from a specific session
    """
    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
               'Right Reward Prob','Left Reward Prob','Reward Given',
              'center_frame','decision_frame']

    root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

    full_name = session_name + '_trials.csv'

    path_name = os.path.join(root_dir,full_name)

    trial_df = pd.read_csv(path_name,names=columns)
    
    """
    Convert to feature matrix
    """
    
    feature_matrix = bp.create_feature_matrix(trial_df,10,mouse_id,session_name,feature_names='Default',imaging=True)
    
    """
    Define function to get frames based on one or two conditions
    """
    
    def extract_frames(df, cond1_name, cond1=False, cond2_name=False, cond2=False, frame_type='decision_frame'):
        if type(cond2_name)==str:
            frames = (df[((df[cond1_name] == cond1) 
                        & (df[cond2_name] == cond2))][frame_type])
            return frames
        else:
            frames =(df[(df[cond1_name] == cond1)][frame_type])
            return frames
        
    """
    Set the parameters to input into extract_frames function
    """
    
    cond1_name = cond1
    cond1_a = 0
    cond1_b = 1
    cond2_name = cond2
    cond2_a = 0
    cond2_b = 1
    
    """
    Extract the frames for the specified conditions and create arrays containing frames 
    for beginning and end of window of interest
    """
    
    # for both outcomes of first condition, with first outcome of second condition
    frames_center_1a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_a, 'center_frame')
    frames_decision_1a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_a, 'decision_frame')

    frames_center_1b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_a, 'center_frame')
    frames_decision_1b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_a, 'decision_frame')

    start_stop_times_1a = [[frames_center_1a - extension], [frames_decision_1a + extension]] 
    start_stop_times_1b = [[frames_center_1b - extension], [frames_decision_1b + extension]]
    
    # for both outcomes of first condition, with second outcome of second condition
    frames_center_2a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_b, 'center_frame')
    frames_decision_2a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_b, 'decision_frame')

    frames_center_2b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_b, 'center_frame')
    frames_decision_2b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_b, 'decision_frame')

    start_stop_times_2a = [[frames_center_2a - extension], [frames_decision_2a + extension]] # start 10 frames before center poke
    start_stop_times_2b = [[frames_center_2b - extension], [frames_decision_2b + extension]] # start 10 frames before center poke

    """
    Remove any neurons that have NaNs in Neuron.C_raw
    """
    
    #plt.plot(neuron.C_raw[0, preStart:trialDecision])
    nNeurons = neuron.C.shape[0]

    # remove neurons that have NaNs
    nan_neurons = np.where(np.isnan(neuron.C_raw))[0]
    nan_neurons = np.unique(nan_neurons)
    good_neurons = [x for x in range(0, nNeurons) if x not in nan_neurons]

    nNeurons = len(good_neurons) # redefine number of neurons
    nTrials_1 = [len(start_stop_times_1a[0][0]), len(start_stop_times_1b[0][0])] # number of trials
    nTrials_2 = [len(start_stop_times_2a[0][0]), len(start_stop_times_2b[0][0])] # number of trials
    
    """
    Calculate window length to include longest trial for each set of conditions
    """
    
    # iterate through to determine duration between preStart and postDecision for each trial
    window_length_1a = []
    window_length_1b = []
    for i in range(0,nTrials_1[0]):
        window_length_1a.append((start_stop_times_1a[1][0].iloc[i] - start_stop_times_1a[0][0].iloc[i]))
    for i in range(0,nTrials_1[1]):
        window_length_1b.append((start_stop_times_1b[1][0].iloc[i] - start_stop_times_1b[0][0].iloc[i]))

    # find longest window between preStart and postDecision and set as length for all trials
    max_window_1 = int([max((max(window_length_1a), max(window_length_1b)))][0])

    # iterate through to determine duration between preStart and postDecision for each trial
    window_length_2a = []
    window_length_2b = []
    for i in range(0,nTrials_2[0]):
        window_length_2a.append((start_stop_times_2a[1][0].iloc[i] - start_stop_times_2a[0][0].iloc[i]))
    for i in range(0,nTrials_2[1]):
        window_length_2b.append((start_stop_times_2b[1][0].iloc[i] - start_stop_times_2b[0][0].iloc[i]))

    # find longest window between preStart and postDecision and set as length for all trials
    max_window_2 = int([max((max(window_length_2a), max(window_length_2b)))][0])

    """
    Pull out frame # for first and last frame in window for each trial, aligned to center poke
    """
    
    start_stop_times_1 = [start_stop_times_1a, start_stop_times_1b]
    aligned_start_1 = np.zeros((np.max(nTrials_1), 2, 2))
    aligned_decision_1 = np.zeros((np.max(nTrials_1), 2, 2))

    start_stop_times_2 = [start_stop_times_2a, start_stop_times_2b]
    aligned_start_2 = np.zeros((np.max(nTrials_2), 2, 2))
    aligned_decision_2 = np.zeros((np.max(nTrials_2), 2, 2))

    
    for i in [0,1]:
        for iTrial in range(0,nTrials_1[i]):
            aligned_start_1[iTrial, 0, i] = start_stop_times_1[i][0][0].iloc[iTrial]
            aligned_start_1[iTrial, 1, i] = start_stop_times_1[i][0][0].iloc[iTrial]+max_window_1
            aligned_decision_1[iTrial,0, i] = start_stop_times_1[i][1][0].iloc[iTrial]-max_window_1
            aligned_decision_1[iTrial, 1, i] = start_stop_times_1[i][1][0].iloc[iTrial]
        
        for iTrial in range(0,nTrials_2[i]):
            aligned_start_2[iTrial,0, i] = start_stop_times_2[i][0][0].iloc[iTrial]
            aligned_start_2[iTrial,1, i] = start_stop_times_2[i][0][0].iloc[iTrial]+max_window_2
            aligned_decision_2[iTrial,0, i] = start_stop_times_2[i][1][0].iloc[iTrial]-max_window_2
            aligned_decision_2[iTrial, 1, i] = start_stop_times_2[i][1][0].iloc[iTrial]
            
    return neuron.C_raw, aligned_start_1, aligned_start_2, aligned_decision_1, aligned_decision_2            
            