import ble2lsl
from ble2lsl.devices import muse2016 # Why do I have to import this seperately? This is dumb
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import time
import numpy as np
import time
from functions import plotTimeDomain, fft, plotFreqDomain
from StreamStaff import getStream_info
from threads import runFunc
import threading
import queue


dummy_streamer = ble2lsl.Dummy(muse2016) #change to Streamer if you want to stream from device
stream_info = resolve_byprop("type", "EEG") #getStream_info(dummy_streamer)
stream = stream_info[0]
psd = fft(stream, output_stream_name='psd')
print(psd)

inlet = StreamInlet(psd, max_chunklen=129, recover=True)
inlet.open_stream()
electrodeOut = [0,0,0,0]
buffer = []
avgCount = 0
avgLength = 10
avgBuffer = 0.7
tempAvg = 0
avg = 0
print("Calculating average from %d frames." % avgLength)
print("The buffer for the light to stay the same is: %f" % avgBuffer)

while True:
    chunk = inlet.pull_chunk()
    #print(chunk)
    if not (np.size(chunk[0]) == 0): # Check for available chunk
        chunkdata = np.transpose(chunk[0]) # Get chunk data and transpose to be CHANNELS x CHUNKLENGTH
        if np.size(buffer) == 0:
            buffer = chunkdata
        else:
            buffer = np.append(buffer, chunkdata, axis=1)

    if not np.size(buffer) == 0:
        while np.size(buffer,1) > 129:
            data = buffer[:,0:129]
            buffer = buffer[:,129:]

            if (avgCount < avgLength):
                electrodeOut[0] = sum(data[1][8:13])
                electrodeOut[1] = sum(data[2][8:13])
                electrodeOut[2] = sum(data[3][8:13])
                electrodeOut[3] = sum(data[4][8:13])
                avgCount +=1

                avg += sum(electrodeOut)/4
                #avg = avg/2


                print("Calculating average")
            elif (avgCount == avgLength):
                print("Average is: %f" % avg)
                avgCount+=1
            else:
                electrodeOut[0] = sum(data[1][8:13])
                electrodeOut[1] = sum(data[2][8:13])
                electrodeOut[2] = sum(data[3][8:13])
                electrodeOut[3] = sum(data[4][8:13])
                tempAvg += sum(electrodeOut)/4
                #tempAvg = tempAvg/2
                
                if (tempAvg < (avg-avgBuffer)):
                    print("Lower average: Making more red")
                elif (tempAvg > (avg+avgBuffer)):
                    print("Higher average: Making more blue")

                else:
                    print("Within average - staying the same")

            

inlet.close_stream()
dummy_streamer.stop()
