# Raymond Klucik
# March 6th 2016 

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style 

style.use('ggplot')

xaxis, xAccel, y, z, a = np.loadtxt('5_sec_jumping_front.csv', delimiter = ',', unpack = True)

def graph():
	#plt.plot(xaxis,xAccel)
	plt.plot(xaxis,y)
	#plt.plot(xaxis,z)
  #plt.plot(xaxis,a)


	plt.title('Accelerometer Data Collected with SensorLog App on iPhone 6S')
	plt.xlabel('time')
	plt.ylabel('Acceleration')

	plt.show()

graph()
