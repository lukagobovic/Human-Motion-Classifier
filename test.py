import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style 
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import FunctionTransformer

style.use('ggplot')

dataset = pd.read_csv("5_sec_walking_front.csv")
data = dataset.iloc[0:,1:4]
# print(data.isna().sum())


mean = data.rolling(31).mean().dropna()
std = data.rolling(31).std().dropna()
max_val = data.rolling(31).max().dropna()
min_val = data.rolling(31).min().dropna()

features = {'mean': mean, 'std': std, 'max': max_val, 'min': min_val}
print(features)

#removing outliers by using percentiles. 
#This essentially will remove all points that are outside the lower 25% and upper 75%
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]

print(data)

window_size = 5
data5 = data.rolling(window_size, center=True).mean()

window_size = 31
data31 = data.rolling(window_size, center=True).mean()

window_size = 51
data51 = data.rolling(window_size, center=True).mean()

# fig, ax = plt.subplots()

plt.plot(std.iloc[0:,1], label = 'Unfiltered')
#ax.plot(data5, label='Window size 5')
#ax.plot(data31, label='Window size 31')
#ax.plot(data51, label='Window size 51')

# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
# ax.legend()

plt.show()

