
from generic_BCI import BCI
from preprocessing import threshold_clf
from classification_tools import get_channels, softmax_predict, encode_ohe_prediction, decode_prediction, DEVICE_SAMPLING_RATE
import biosppy.signals as bsig 
import numpy as np
import ble2lsl
from ble2lsl.devices import muse2016 
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import time
from functions import plotTimeDomain, fft, plotFreqDomain
#from StreamStaff import getStream_info

#from threads import runFunc
#import threading
#import queue
from phue import Bridge #HUE 


b = Bridge('192.168.0.154')  #192.168.2.116 (home) - 192.168.0.154 (nates top)
b.connect()
lights = b.get_light_objects()
k = 5
k=k/10
for light in lights:
    light.xy = [0.1000, k]
print("Light default set")


def transform(buffer):
    '''gets some some choice of channels, epochs on some length, get power features in alpha_high band'''
    #parameters
    epoch_len = 250
    channels = ['AF7', 'AF8']
    device = 'muse'
    #end of parameters
    
    sr = DEVICE_SAMPLING_RATE['muse']
    
    # get the latest epoch_len samples from the buffer
    transformed_signal = np.array(buffer[-epoch_len:, :])
    
    # get the selected channels
    transformed_signal = get_channels(transformed_signal, channels, device)
    
    # get power features of signal
    ts, theta, alpha_low, alpha_high, beta, gamma = bsig.eeg.get_power_features(transformed_signal,
                                                                                sampling_rate=sr, size=epoch_len/sr)    
    
    transformed_signal = alpha_high
    
    return transformed_signal # return alpha_high power

def clf(input_):
    '''classifies based on power channel-wise and returns string of class'''
    threshold = 0.07 # arbitrarily chosen atm, choose one that works
    print(input_)
    class_bool = threshold_clf(input_, threshold, clf_consolidator='all')
    
    pred = decode_prediction(class_bool, {False: 'Non-concentrated', True: 'Concentrated'})
    
    return pred

def changeLight(input_):
    global k
    if (input_ == 'Non-concentrated'):
        print(input_)
        if (k >= 9):
            k = 9.0
        else:
            k = k + .5
    else:
        print(input_)
        if (k <= 1):
            k = 1.0
        else:
            k = k - .5

    k = k/10 #put it into proper format so it works w light funct
    for light in lights:
        light.xy = [0.1000, k]
    k = k * 10

stream_info =  resolve_byprop("type", "EEG")
stream = stream_info[0]
inlet = StreamInlet(stream, recover=True)

BCI(inlet, clf, transform, action=changeLight, buffer_length=512, n_channels=5)


""" Todo: 
- Calculated threshold value per individual - check out BCI workshop
- incorporate the blink classifier - Where can we find the blink classifier?
- pip installable the extra tools (from AI team)
- clean and document (requirements.txt)"""