from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

def normalizeData(data, windowSize, mean=None, std=None):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))] = np.nan
    data.interpolate(method='linear', inplace=True)
    data = data.rolling(windowSize, center=True).mean()
    data.fillna(method='ffill', inplace=True)
    if mean is not None and std is not None:
        dataNew = (data - mean) / std
    else:
        dataNew = (data - data.mean()) / data.std()
    return dataNew

def extract_features(data, wsize):
    features = []
    xyz_data = data.iloc[:, 1:4].rolling(window=wsize, center=True)
    features.append(xyz_data.mean())
    features.append(xyz_data.std())
    features.append(xyz_data.max())
    features.append(xyz_data.min())
    features.append(xyz_data.median())
    features.append(xyz_data.var())
    features.append(xyz_data.kurt())
    features.append(xyz_data.skew())
    features.append(xyz_data.max()-xyz_data.min())
    features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
                               xyz_data.min(), xyz_data.median(), xyz_data.var(),  
                               xyz_data.kurt(), xyz_data.skew(),(xyz_data.max()-xyz_data.min())
                               ))
    np.random.shuffle(features)
    datFrame = pd.DataFrame(features)
    return datFrame
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

fileName = 'BennettRawDataWalking'

# Load the feature data
originalData = pd.read_csv('MemberData/'+fileName+'.csv')
mean = originalData.mean()
std = originalData.std()
normalizedData = normalizeData(originalData, 10, mean, std)
tempData = pd.DataFrame()
for j in range(0, len(normalizedData) - 500, 500):
    df = extract_features(normalizedData.iloc[j:j+500-1, :], 10)
    tempData = pd.concat([tempData, df])
tempData = tempData.fillna(tempData.mean())
X = tempData.iloc[:,0:18]
y_pred = model.predict(X)

# Make predictions
predicted_class = np.bincount(y_pred.astype(int)).argmax()
if predicted_class == 0:
    print("This CSV contains walking data.")
    originalData['Walking(0)/Jumping(1)'] = 0
    originalData.to_csv('OutputData/'+fileName+'.csv', index=False)
else:
    print("This CSV contains jumping data.")
    originalData['Walking(0)/Jumping(1)'] = 1
    originalData.to_csv('OutputData/'+fileName+'.csv', index=False)