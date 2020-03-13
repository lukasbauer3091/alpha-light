import numpy as np
import biosppy.signals as bsig

DEVICE_SAMPLING_RATE = {'muse': 256, # is this right? is it 220 Hz (see documentation)?
                       }


def get_channels(signal, channels, device='muse'):
    """
    Returns a signal with only the desired channels.
    
    Arguments:
        signal: a signal of shape [n_samples, n_channels]
        channels: an array of the str names of the desired channels. returned in this order.
        device: str name of the device. 
    
    Returns:
        numpy array of signal with shape [n_channels, n_desired_channels]. 
        Includes only the selected channels in the order given.
    """
    
    # check device; each device has its own ch_ind dictionary corresponding to its available channels
    if device == 'muse':
        ch_ind_muse = {'TP9': 0, 'AF7': 1, 'AF8': 2, 'TP10': 3}
        return_signal = np.array([signal[:, ch_ind_muse[ch]] for ch in channels]).T
    
    
    return return_signal

def transform(buffer, epoch_len, channels, device='muse', filter_=False, filter_kwargs={}):
    """
    Ensemble transform function. Takes in buffer as input. Extracts the appropriate channels and samples. Performs filtering.

    Arguments:
        buffer: the latest stream data. shape: [n_samples, n_channels]
        epoch_len: the length of epoch expected by predictor in number of samples.
        channels: list of channels expected by predictor. See get_channels.
        device: string of device name. used to get channel and sampling_rate information
        filter_: boolean of whether to perform filtering
        filter_kwargs: dictionary of kwargs to be passed to filtering function. See biosppy.signals.tools.filter_signal.
            by default, an order 8 bandpass butter filter is performed between 2Hz and 40Hz.
    """

    # get the latest epoch_len samples of the buffer
    transformed_signal = np.array(buffer[-epoch_len:, :])
    
    # get the selected channels
    transformed_signal = get_channels(transformed_signal, channels, device)
    
    #filter_signal 
    if filter_:
        # create dictionary of kwargs for filter_signal
        filt_kwargs = {'sampling_rate': DEVICE_SAMPLING_RATE[device], 
                       'ftype': 'butter',
                       'band': 'bandpass', 
                       'frequency': (2, 40),
                       'order': 8}
        filt_kwargs.update(filter_kwargs)
        
        
        transformed_signal, _, _ = bsig.tools.filter_signal(signal=transformed_signal.T, **filt_kwargs)
        transformed_signal = transformed_signal.T
    
    return transformed_signal

def softmax_predict(input_, predictor, thresh=0.5):
    """
    Consolidates a softmax prediction to a one-hot encoded prediction.
    
    Arguments:
        input_: the input taken by the predictor
        predictor: function which returns a softmax prediction given an input_
        thresh: the threshold for a positive prediction for a particular class.
    
    """
    
    pred = np.array(predictor(input_))
    
    return (pred >= thresh).astype(int)


def encode_ohe_prediction(prediction):
    '''Returns the index number of the positive class in a one-hot encoded prediction.'''
    return np.where(np.array(prediction) == 1)[0][0]


def decode_prediction(prediction, decode_dict):
    '''Returns a more intelligible reading of the prediction based on the given decode_dict'''
    return decode_dict[prediction]


