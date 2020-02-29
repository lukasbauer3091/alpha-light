#!/usr/bin/python
#Olivia
from phue import Bridge
import random

b = Bridge('192.168.0.154') # Enter bridge IP here.

#If running for the first time, press button on bridge and run with b.connect() uncommented
b.connect()

lights = b.get_light_objects()

for k in range(0,10):
	k = k/10
	for light in lights:
		light.xy = [0.1000, k] 
		print (light.xy)
		

#going from [0.1, 0.1->0.9] turns from blue to green