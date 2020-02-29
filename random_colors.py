#!/usr/bin/python
#by Arhunna
from phue import Bridge
import random

b = Bridge('192.168.0.154') # Enter bridge IP here.

#If running for the first time, press button on bridge and run with b.connect() uncommented
b.connect()

lights = b.get_light_objects()
#lights.xy = [0.1000, 0.1000]

k = 0 

for i in range(100):
	val = input("enter 'b' for blue-er and 'g' for green-er")

	if (val == 'b'): #check what value is entered
		if (k <= 1):
			k = 1
		else:
			k = k - 1	
	elif (val == 'g'): 
		if (k >= 9):
			k = 9
		else:
			k = k + 1
			
	k = k/10 #put it into proper format so it works w light funct

	print (k)
	for light in lights:
		#light.xy = [random.random(),random.random()]
		light.xy = [0.1000, k]
		#light.xy = [0.8524, 0.2154]
		print (light.xy)
		k = k * 10

#light.xy = [0.8524, 0.2154] is orange

#going from [0.1, 0.1->0.9] turns from blue to green