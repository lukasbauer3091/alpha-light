"""Module containing functions for the preparation of EEG data for use with ML models and general BCI purposes.

    Usage Example: [TODO]
"""

import numpy as np
from scipy import stats
import biosppy.signals as bsig

import csv
import math as m

def rowskip(data):
    """
    Determines the number of lines to skip as start of csv for EEG data_stim

    Arguments:
        data: csv file name of the labels

    Returns:
        Appropriate value to pass for skiprows in np.loadtxt
    """
    with open(data, newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)
    return int(row1[1])+2


def get_corrupt(data):
    """
    Obtains the start and end timestamps for each corrupt period as ints

    Arguments:
            data: csv file name for labels

    Returns:
            corrupt: array of pairs of indeces denoting start and end of corruption
    """
    with open(data, newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)

        corrupt = []
        for i in range(int(row1[1])):
            corrupt.append(next(reader))

        #There has to be a better way to accomplish what I'm about to do
        corrupt = [list(map(float, item)) for item in corrupt]  #string to flot
        corrupt = [np.floor(entry) for entry in corrupt]  #float floored
        corrupt = [list(map(int, item)) for item in corrupt]  #flot to int
        return corrupt


def get_corrupt_indeces(label_file, sr):
    """
    interpolates from start-end timestamps to obtain an array of every index
    which corresponds to a corrupted instance

    Arguments:
            label_file: csv filename for labels for a single subject
    """
    filledin = np.array([])
    bad = get_corrupt(label_file)
    for i in range(len(bad)):
        start = m.floor(bad[i][0]*sr)
        stop = m.floor(bad[i][1]*sr)
        elong = (np.linspace(start, stop, stop-start))
        filledin = np.concatenate((filledin, elong))
    return filledin.astype(int)


def epoch(signal, window_size, inter_window_interval):
    """
    Creates overlapping windows/epochs of EEG data from a single recording.

    Arguments:
        signal: array of timeseries EEG data of shape [n_samples, n_channels]
        window_size(int): desired size of each window in number of samples
        inter_window_interval(int): interval between each window in number of samples (measured start to start)

    Returns:
        numpy array object with the epochs along its first dimension
    """

    #Calculate the number of valid windows of the specified size which can be retrieved from data
    num_windows = 1 + (len(signal) - window_size) // inter_window_interval

    epochs = []
    for i in range(num_windows):
        epochs.append(signal[i*inter_window_interval:i*inter_window_interval + window_size])

    return np.array(epochs)


def labels_from_timestamps(timestamps, sampling_rate, length):
    """takes an array containing timestamps (as floats) and
    returns a labels array of size 'length' where each index
    corresponding to a timestamp via the 'samplingRate'.

    Arguments:
        timestamps: an array of floats containing the timestamps for each event (units must match sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        length(int): the number of samples of the corresponing EEG run.

    Returns:
        an integer array of size 'length' with a '1' at each time index where a corresponding timestamp exists, and a '0' otherwise.
    """
    #TODO: function currently assumes binary labels. Generalize to arbitrary labels.


    #create labels array template
    labels = np.zeros(length)

    #calculate corresponding index for each timestamp
    labelIndices = np.array(timestamps * sampling_rate)
    labelIndices = labelIndices.astype(int)

    #flag label at each index where a corresponding timestamp exists
    labels[labelIndices] = 1
    labels = labels.astype(int)

    return np.array(labels)


def label_epochs(labels, window_size, inter_window_interval, label_method):
    """create labels for individual eoicgs of EEG data based on the label_method.
    Note: For now, the "containment" and "count" label_methods assume binary labels.

    Arguments:
        labels: an integer array indicating a class for each sample measurement
        window_size(int): size of each window in number of samples (matches window_size in EEG data)
        inter_window_interval(int): interval between each window in number of samples (matches inter_window_interval in EEG data)
        label_method(str/func): method of consolidating labels contained in epoch into a single label:
            "containment": whether a positive label is contained in the epoch,
            "count": the count of positive labels in the epoch,
            "mode": the most common label in the epoch
            func: simply pass in the custome function you'd like to determine an epoch's label: func_name(epoched_labels)

    Returns:
        a numpy array with a label correponding to each epoch
    """

    # epoch the labels themselves so each epoch contains a label at each sample measurement
    epochs = epoch(labels, window_size, inter_window_interval)

    #if a positive label [1] is occurs in the epoch, give epoch positive label
    if label_method == "containment":
        epoch_labels = [int(1 in epoch) for epoch in epochs]

    elif label_method == "count":
        # counts the number of occurences of the positive label [1]]
        #TODO: maybe have have the label to be counted specified by the function
        #perhaps these should be broken up to different functions?
        epoch_labels = [epoch.count(1) for epoch in epochs]

    elif label_method == "mode":
        #choose the most common label occurence in the epoch and default to the smallest if multiple exist
        epoch_labels = [stats.mode(epoch)[0][0] for epoch in epochs]

    elif callable(label_method):
        epoch_labels = [label_method(epoch) for epoch in epochs]


    return np.array(epoch_labels)


def label_epochs_from_timestamps(timestamps, sampling_rate, length, window_size, inter_window_interval,
                                 label_method="containment"):
    """
    Directly creates labels for individual windows of EEG data from timestamps of events.

    Arguments:
        timestamps: an array of floats containing the timestamps for each event (units must match sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        length(int): the number of samples of the corresponing EEG run.
        window_size(int): size of each window in number of samples (matches window_size in EEG data)
        inter_window_interval(int): interval between each window in number of samples (matches inter_window_interval in EEG data)
        label_method(str): method of consolidating labels contained in window into a single label:
            "containment": whether a positive label is contained in the window,
            "count": the count of positive labels in the window,
            "mode": the most common label in the window

    Returns:
        an array with a label correponding to each window
    """
    labels = labels_from_timestamps(timestamps, sampling_rate, length)

    epoch_labels = label_epochs(labels, window_size, inter_window_interval, label_method)

    return epoch_labels


def epoch_and_label(data, sampling_rate, timestamps, window_size, inter_window_interval, label_method="containment"):
    """
    Epochs a signal (single EEG recording) and labels each epoch using timestamps of events
    and a chosen labelling method.

    Arguments:
        data: array of timeseries EEG data of shape [n_samples, n_channels]
        timestamps: an array of floats containing the timestamps for each event (units must match sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        window_size(float): desired size of each window in units of time (matches sampling_rate)
        inter_window_interval(float): interval between each window in units of time (measured start to start; matches sampling_rate)
        label_method(str): method of consolidating labels contained in window into a single label:
            "containment": whether a positive label is contained in the window,
            "count": the count of positive labels in the window,
            "mode": the most common label in the window


    Returns:
        epochs: an array of the epochs with shape [n_epochs, n_channels]
        labels: an array of the labels corresponding to each epoch of shape [n_epochs, ]
    """

    epochs = epoch(data, int(window_size*sampling_rate), int(inter_window_interval*sampling_rate))
    labels = label_epochs_from_timestamps(timestamps, sampling_rate, len(data),
                                          int(window_size*sampling_rate), int(inter_window_interval*sampling_rate),
                                          label_method=label_method)

    return epochs, labels


def lbl_wo_corrupt(label_file, timestamps, sr, length, window_size, window_step):
    """
    PREAMBLE: takes in label_file to determine corrupt instances,
    uses get_corrupt_indeces function to return every index of a
    corrupt channel, use label_from_timestamps function to elongate
    list of provided timestamps for blinks, make every instance which
    is corrupt a 2, containment to obtain labelled window, get rid of
    windows with title 2
    returns: all non-corrupt windowed-labels, all windowed labels (to be used
    to get rid of corrupt epochs later)

    Arguments:
       label_file: filename for csv of labels
       timestamps: csv of timestamps
       sr: sampling rate
       length: length of eeg channel (i.e number of datum in an eeg channel)
       window_size: number of data points in training input elements
       window_step: increment for "lateral" window shift across all data
    """
    labels = labels_from_timestamps(timestamps, sr, length)
    corrupt_indeces = get_corrupt_indeces(label_file, sr)
    labels[corrupt_indeces] = 2
    windowed_raw = label_epochs(labels, window_size, window_step, max)
    windowed_refined = [window for window in windowed_raw if window != 2]
    return windowed_refined, windowed_raw


def epoch_subject_data(dataset, window_size, window_step, sensor):
    """
    Iterates through large array of subjects and returns one array containing
    unfiltered training data

    Arguments:
          dataset: array of subjects and subject eeg voltages
          window_size, window_step: see above
          sensor: refers to which data channel you wish to generate data from

    Returns:
          subject_epochs: array of all subject epochs for training (unfiltered)
    """
    placeholder = np.array(dataset)
    number_of_subjects = len(dataset) #change this to first value in shape array since one dimensional might just give a shit ton of stuff
    subject_epochs = []

    for i in range(number_of_subjects):
        subject_data = placeholder[i]
        channel_of_interest = subject_data[:,sensor]
        epoched = epoch(channel_of_interest, window_size, window_step)

        for j in range(len(epoched)):
            subject_epochs.append(epoched[j])

    return np.array(subject_epochs)


def epoch_subject_labels(dataset, labels, label_files, sr, window_size, window_step, mode='default'):
    """
    will return the refined labels, raw labels to be used for refining the
    data epochs. takes in the array of subject blink timestamps, NOT individual.

    Arguments:
       dataset: array of eeg data for subjects.  used for determining length parameter in labelling function
       labels: array of timstamps
       label_files: label csv filenames
       mode: 'default' returns raw and refined labelled windows (ie. including corrupt
       vs no corrupt channels). 'only_raw' returns every possible label window regardless
       of corruption

    Returns:
       raw or refined (no corrupt) labelled windows
    """
    number_of_subjects = len(labels)
    placeholder1 = np.array(labels)
    placeholder2 = np.array(dataset)  #need this for the length parameter

    subject_labels_raw = []
    subject_labels_refined = []

    for i in range(number_of_subjects):
        subject_labels = placeholder1[i]
        subject_data = placeholder2[i]
        subject_len = len(subject_data[:,0])
        refined, raw = lbl_wo_corrupt(label_files[i], subject_labels, sr, subject_len, window_size, window_step)

        for j in range(len(raw)):
            subject_labels_raw.append(raw[j])

        for k in range(len(refined)):
            subject_labels_refined.append(refined[k])

    if mode=='only_raw':
        return np.array(subject_labels_raw)

    elif mode =='default':
        return np.array(subject_labels_raw), np.array(subject_labels_refined)


def purify_epochs(epoched_data, raw_epoched_labels):
    """
    takes in the long list of raw epochs and returns only non-corrupt ones
    """
    indeces = np.where(raw_epoched_labels !=2)[0]
    return np.array(epoched_data[indeces])


def compute_signal_std(signal, corrupt_intervals=None, sampling_rate=1):
    """
    Computes and returns the standard deviation of a signal channel-wise (while avoiding corrupt intervals)
    
    Arguments:
        signal: signal of shape [n_samples, n_channels]
        corrupt_intervals: an array of 2-tuples indicating the start and end time of the corrupt interval (units of time)
        sampling_rate: the sampling rate in units of samples/unit of time
        

    Returns:
        standard deviation of signal (computed per channel) of shape [1, n_channels]
    """
    
    # convert signal into numpy array for use of its indexing features
    signal = np.array(signal)
    
    if corrupt_intervals is not None:
        #convert corrupt_indices from units of time to # of samples 
        corrupt_intervals = [(int(cor_start * sampling_rate), int(cor_end * sampling_rate)) for cor_start, cor_end in corrupt_intervals]
        
        #find all indices to keep
        good_indices = []

        # for each index in the signal, check if it is contained in any of the corrupt intervals to
        for ind in range(len(signal)):
            good_indices.append(not np.any([ind in range(cor_start, cor_end) for cor_start, cor_end in corrupt_intervals]))
    
        signal = signal[good_indices]

    # compute and return std on non_corrupt parts of signal
    return np.std(signal, axis=0, dtype=np.float32) 


def split_corrupt_signal(signal, corrupt_intervals, sampling_rate=1):
    """
    Splits a signal with corrupt intervals and returns array of signals with the corrupt intervals filtered out.
    This is useful for treating each non_corrupt segment as a seperate signal to ensure continuity within a single signal. 
    
    Arguments:
        signal: signal of shape [n_samples, n_channels]
        corrupt_intervals: an array of 2-tuples indicating the start and end time of the corrupt interval (units of time)
        sampling_rate: the sampling rate in units of samples/unit of time
        

    Returns:
        array of non_corrupt signals of shape [n_signal, n_samples, n_channels]
    """
    
    #convert corrupt_indices from units of time and flatten into single array
    corrupt_intervals = np.array(corrupt_intervals * sampling_rate).flatten()

        
    # convert signal into numpy array for use with library
    signal = np.array(signal)
    
    #split signal on each cor_start or cor_end and discard the regions in between cor_start and cor_end
    signals = np.split(signal, corrupt_intervals)[::2]
    
    return signals


def epoch_band_features(epoch, sampling_rate, bands='all'):
    """
    Computes power features of EEG frequency bands over entire epoch channel-wise.

    Arguments:
        epoch: a single epoch of shape [n_samples, n_channels]
        sampling_rate: the sampling rate of the signal in units of samples/unit of time
        bands: the requested frequency bands to get power features for.
            'all': all of ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
            otherwise an array of strings of the desired bands.

    Returns:
        a dictionary of arrays of shape [1, n_channels] containing the power features over each band per channel.
    """

    if bands == 'all':
        bands = ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']


    # computes length of epoch to compute band power features over entire epoch
    w_size = np.array(epoch).shape[0] / sampling_rate

    # compute power for each band
    _, theta, alpha_low, alpha_high, beta, gamma = bsig.eeg.get_power_features(epoch, sampling_rate=sampling_rate, size=w_size, overlap=0)

    # initialize return dict
    band_features = {'theta': theta, 
                    'alpha_low': alpha_low,
                    'alpha_high': alpha_high,
                    'beta': beta,
                    'gamma': gamma }

    # return only requested bands (intersection of bands available and bands requested)                 
    band_features = {band: band_features[band] for band in bands & band_features.keys()}

    return band_features

def threshold_clf(features, threshold, clf_consolidator='any'):
  """
    Classifies given features based on a given threshold.
    
    Arguments:
        features: an array of numerical features to classify
        threshold: the threshold for classification. A single number, or an array corresponding to `features` for element-wise comparison.
        clf_consolidator: method of consolidating element-wise comparisons with threshold into a single classification.
            'any': positive class if any features passes the threshold
            'all': positive class only if all features pass threshold
            'sum': a count of the number of features which pass the threshold
            function: a custom function which takes in an array of booleans and returns a consolidated classification
    
    Returns:
        A classification for the given features. Return type `clf_consolidator`.
  """

  try:
    label = np.array(features) > np.array(threshold)
  except ValueError as v_err:
    print("Couldn't perform comparison between features and thresholds. Try a different format for the threshold.")
    raise v_err
  

  if clf_consolidator == 'any': 
    label = np.any(label)

  elif clf_consolidator == 'all':
    label = np.all(label)

  elif clf_consolidator == 'sum':
    label = np.sum(label)

  elif callable(clf_consolidator):
    try:
      label = clf_consolidator(label)
    except TypeError as t_err:
      print("Couldn't use clf_consolidator function to consolidate classification")
      raise t_err
  
  return label