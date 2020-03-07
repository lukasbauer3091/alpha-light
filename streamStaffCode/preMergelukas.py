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
from phue import Bridge #HUE 

b = Bridge('192.168.0.154') 
b.connect()
lights = b.get_light_objects()
k = 0

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
avgBuffer = 0.4
tempAvg = 0
avg = 0
avgArr = []

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

                avgArr.append(sum(electrodeOut)/4)  
                print("Still calculating")
                avgCount+=1
            
            
            elif (avgCount == avgLength):
                avg = sum(avgArr)/len(avgArr)
                print("Average is: %f" % avg)
                avgCount +=1
                
            else:
                electrodeOut[0] = sum(data[1][8:13])
                electrodeOut[1] = sum(data[2][8:13])
                electrodeOut[2] = sum(data[3][8:13])
                electrodeOut[3] = sum(data[4][8:13])
                tempAvg = sum(electrodeOut)/4
                print(tempAvg)
                
                #going from [0.1, 0.1->0.9] turns from blue to green
                
                if (tempAvg < (avg-avgBuffer)):
                    print("Lower average: Making more green")
                    if (k >= 9):
                        k = 9
                    else:
                        k = k + 1
                elif (tempAvg > (avg+avgBuffer)):
                    print("Higher average: Making more blue")
                    if (k <= 1):
                        k = 1 
                    else:
                        k = k - 1
                else:
                    print("Within average - staying the same")

                k = k/10 #put it into proper format so it works w light funct
                light.xy = [0.1000, k]
                k = k * 10

inlet.close_stream()
dummy_streamer.stop()
