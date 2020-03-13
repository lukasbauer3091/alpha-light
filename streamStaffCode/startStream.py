import ble2lsl
from ble2lsl.devices import muse2016 # Why do I have to import this seperately? This is dumb
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import time
import numpy as np
import time
from functions import plotTimeDomain, fft, plotFreqDomain
from phue import Bridge #HUE 

"""
b = Bridge('192.168.0.154') 
b.connect()
lights = b.get_light_objects()
k = 5
"""


dummy_streamer = ble2lsl.Streamer(muse2016) #change to Streamer if you want to stream from device