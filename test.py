# Raymond Klucik
# March 6th 2016 

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style 

style.use('ggplot')

xaxis, x, y, z, a = np.loadtxt('Raw Data walking jacket.csv', delimiter = ',', unpack = True)

def graph():
	plt.plot(xaxis,x)
	#plt.plot(xaxis,y)
	#plt.plot(xaxis,z)
	#plt.plot(xaxis,a)


	plt.title('Accelerometer Data Collected with SensorLog App on iPhone 6S')
	plt.ylabel('')
	plt.xlabel('7 Everyday Activites')

	plt.show()

graph()
