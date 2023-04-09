from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import h5py
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import skew
import pickle



df1 = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv',nrows = 30005)
df2 = pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv',nrows = 30005)
df3 = pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv',nrows = 30005)

df4 = pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv',nrows = 30005)
df5 = pd.read_csv('MemberData/CJRawDataJacketWalking.csv',nrows = 30005)
df6 = pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv',nrows = 30005)
listOfWalkingData = pd.DataFrame()
listOfWalkingData.concat([df1,df2,df3,df4,df5,df6])
listOfWalkingData['labels'] = 0
print(listOfWalkingData)
listOfJumpingData = pd.DataFrame()

df7 = pd.read_csv('MemberData/LukaRawDataJumping.csv')

# BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')

df8 = pd.read_csv('MemberData/CJRawDataJumping.csv')


listOfJumpingData.concat([df7,df8])
listOfJumpingData['labels'] = 1
print(listOfJumpingData)


lisfOfCombinedData = listOfWalkingData + listOfJumpingData
print(listOfJumpingData)

LukaWalkingData = pd.concat(
    [df1,df2,df3]
)
CJWalkingData = pd.concat(
     [df4,df5,df6]
 )

all_data = {
    'Luka': {'walking': LukaWalkingData, 'jumping': df7},
    # 'Bennett': {'walking': BennettWalkingData, 'jumping': BennettJumpingData},
    'CJ': {'walking': CJWalkingData, 'jumping': df8}
}


#use a window size of like 5-15
def normalizeData(data,windowSize):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))] = np.nan

    data.interpolate(method = 'linear',inplace=True)

    data = data.rolling(windowSize,center = True).mean()

    data.fillna(method = 'ffill',inplace=True)

    data = data.loc[5:,:]

    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dataNew = pd.DataFrame(x_scaled)

    return dataNew

normalizedData = []
for i in range(0,8):
    normalizedData.append(normalizeData(lisfOfCombinedData[i],10))

def extract_features_walking(data, wsize):
    features = []
    xyz_data = data.iloc[:, 1:4].rolling(window=wsize)
    features.append(xyz_data.mean())
    features.append(xyz_data.std())
    features.append(xyz_data.max())
    features.append(xyz_data.min())
    features.append(xyz_data.median())
    features.append(xyz_data.var())
    features.append(xyz_data.kurt())
    features.append(xyz_data.skew())
    features.append(xyz_data.max()-xyz_data.min())

    # Append a column of zeros to the combined X, Y, and Z data
    zeros = np.zeros((xyz_data.mean().shape[0], 1))
    features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
                               xyz_data.min(), xyz_data.median(), xyz_data.var(),  
                               xyz_data.kurt(), xyz_data.skew(),(xyz_data.max()-xyz_data.min())
                               , zeros))

    np.random.shuffle(features)
    datFrame = pd.DataFrame(features)
    return datFrame

def extract_features_jumping(data, wsize):
    features = []
    xyz_data = data.iloc[:, 1:4].rolling(window=wsize)
    features.append(xyz_data.mean())
    features.append(xyz_data.std())
    features.append(xyz_data.max())
    features.append(xyz_data.min())
    features.append(xyz_data.median())
    features.append(xyz_data.var())
    features.append(xyz_data.kurt())
    features.append(xyz_data.skew())
    features.append(xyz_data.max()-xyz_data.min())
   
    # Append a column of zeros to the combined X, Y, and Z data
    ones = np.ones((xyz_data.mean().shape[0], 1))
    features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
                               xyz_data.min(), xyz_data.median(), xyz_data.var(),  
                               xyz_data.kurt(), xyz_data.skew(),(xyz_data.max()-xyz_data.min())
                               , ones))
    np.random.shuffle(features)
    datFrame = pd.DataFrame(features)
    return datFrame

featureData = pd.DataFrame()
tempData = pd.DataFrame()
for i in range(0,6):
    for j in range(0, len(normalizedData[i]) - 500, 500):
        df = extract_features_walking(normalizedData[i].iloc[j:j+500-1, :],10)
        tempData = pd.concat([tempData,df])

featureData = pd.concat([featureData,tempData]) 
tempData = pd.DataFrame()
for i in range(6,8):
   
    for j in range(0, len(normalizedData[i]) - 500, 500):
        df = extract_features_jumping(normalizedData[i].iloc[j:j+500-1, :],10)
        tempData = pd.concat([tempData,df])
featureData = pd.concat([featureData,tempData])
featureData.interpolate(method = 'linear',inplace=True)


X = featureData.iloc[:,0:27]
y = featureData.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle = True, random_state=0)

with h5py.File('./ProjectFileTest.hdf5', 'w') as f:
    # Create sub groups for each member
    for member_name, member_data in all_data.items():
        member_group = f.create_group(member_name)
        member_group.create_dataset('walking', data=member_data['walking'])
        member_group.create_dataset('jumping', data=member_data['jumping'])

    # Create a sub group for the dataset
    dataset_group = f.create_group('dataset')


    train_segments = pd.concat([X_train,y_train],axis=1)
    test_segments = pd.concat([X_test,y_test],axis=1)

    dataset_group.create_dataset('train',data=train_segments.values, dtype=train_segments.values.dtype)
    dataset_group.create_dataset('test',data=test_segments.values, dtype=test_segments.values.dtype)

