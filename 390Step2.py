import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import h5py

#Read in raw data, note, number of rows has a 4 at the end due to the rolling window,
# which will truncate the first 4 values in the list. Each row number is also a multiple of 500
# to ensure no overlap occurs when segmenting
df1 = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv')
df2 = pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv')
df3 = pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv')

df4 = pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv')
df5 = pd.read_csv('MemberData/CJRawDataJacketWalking.csv')
df6 = pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv')

df7 = pd.read_csv('MemberData/BennettRawDataBackPocketWalking.csv')
df8 = pd.read_csv('MemberData/BennettRawDataFrontPocketWalking.csv')
df9 = pd.read_csv('MemberData/BennettRawDataJacketWalking.csv')

df10 = pd.read_csv('MemberData/LukaRawDataFrontPocketJumping.csv')
df11 = pd.read_csv('MemberData/LukaRawDataJacketJumping.csv')
df12 = pd.read_csv('MemberData/LukaRawDataBackPocketJumping.csv')

df13 = pd.read_csv('MemberData/BennettRawDataBackPocketJumping.csv')
df14 = pd.read_csv('MemberData/BennettRawDataJacketJumping.csv')
df15 = pd.read_csv('MemberData/BennettRawDataFrontPocketJumping.csv')

df13 = df13.iloc[::2, :]
df14 = df14.iloc[::2, :]
df15 = df15.iloc[::2, :]

df16 = pd.read_csv('MemberData/CJRawDataFrontPocketJumping.csv')
df17 = pd.read_csv('MemberData/CJRawDataBackPocketJumping.csv')
df18 = pd.read_csv('MemberData/CJRawDataJacketJumping.csv')

LukaWalkingData = pd.concat(
    [df1,df2,df3]
)
CJWalkingData = pd.concat(
     [df4,df5,df6]
 )

BennettWalkingData = pd.concat(
   [df7,df8,df9]
)

LukaJumpingData = pd.concat(
   [df10,df11,df12]
)
BennettJumpingData = pd.concat(
   [df13,df14,df15]
)
CJJumpingData = pd.concat(
   [df16,df17,df18]
)

all_data = {
    'Luka': {'walking': LukaWalkingData, 'jumping': LukaJumpingData},
    'Bennett': {'walking': BennettWalkingData,'jumping':BennettJumpingData },
    'CJ': {'walking': CJWalkingData, 'jumping': CJJumpingData}
}

listOfWalkingData = [df1,df2,df3,df4,df5,df6,df7,df8,df9]

listOfJumpingData = [df10,df11,df12,df13,df14,df15,df16,df17,df18]

for i in range(len(listOfWalkingData)):
  # Remove outliers
  window_size = 5
  df = listOfWalkingData[i].iloc[:,1:]
  df = df.iloc[:-(df.shape[0] % 500)+(window_size-1)] 
  Q1 = df.quantile(0.30)
  Q3 = df.quantile(0.70)
  IQR = Q3 - Q1
  df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]

  #Interpolate the missing values
  df.interpolate(method='linear', inplace=True)
  
  #Rolling average filter
  df = df.rolling(window_size).mean().dropna()
    
  # Normalize the data
  scaler = MinMaxScaler()
  df = scaler.fit_transform(df)
  df = pd.DataFrame(df)
  listOfWalkingData[i] = df

#Create a dataframe with all new filtered data
listOfWalkingDataDF = pd.DataFrame()
for df in listOfWalkingData:
  listOfWalkingDataDF = pd.concat([listOfWalkingDataDF,df])

#Add a label column to classify it as walking data
listOfWalkingDataDF['activity'] = 0  


for i in range(len(listOfJumpingData)):
  # Remove outliers
  window_size = 5
  df = listOfJumpingData[i].iloc[:,1:]
  df = df.iloc[:-(df.shape[0] % 500)+(window_size-1)]   
  Q1 = df.quantile(0.30)
  Q3 = df.quantile(0.70)
  IQR = Q3 - Q1
  df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]

  #Interpolate the missing values
  df.interpolate(method='linear', inplace=True)
  
  #Rolling average filter
  df = df.rolling(window_size).mean().dropna()
    
  # Normalize the data
  scaler = MinMaxScaler()
  df = scaler.fit_transform(df)
  df = pd.DataFrame(df)
  listOfJumpingData[i] = df

#Create a dataframe with all new filtered data
listOfJumpingDataDF = pd.DataFrame()
for df in listOfJumpingData:
  listOfJumpingDataDF = pd.concat([listOfJumpingDataDF,df])

#Add a label column to classify it as jumping data
listOfJumpingDataDF['activity'] = 1 

lisfOfCombinedData = pd.concat([listOfWalkingDataDF,listOfJumpingDataDF])
data = lisfOfCombinedData

# Define window size 
window_size = 5

# Segment the data into windows of a given size
def segment_data(data, window_size):
  num_samples = int(np.floor(data.shape[0] / (window_size * 100)))  # 100 Hz sampling rate
  segments = []
  for i in range(num_samples):
    start_idx = i * window_size * 100
    end_idx = start_idx + window_size * 100
    segment = data.iloc[start_idx:end_idx]
    segmentNP = segment.to_numpy()
    np.random.shuffle(segmentNP)
    segment = pd.DataFrame(segmentNP)
    segments.append(segment)
  return segments
# Segment the data and shuffle it
segments = segment_data(data, window_size)
np.random.shuffle(segments)

# Split the data into training and testing sets
train_size = int(np.floor(len(segments) * 0.9))
train_segments = segments[:train_size]
test_segments = segments[train_size:]

with h5py.File('./ProjectFile.hdf5', 'w') as f:
    # Create sub groups for each member
    for member_name, member_data in all_data.items():
        member_group = f.create_group(member_name)
        member_group.create_dataset('walking', data=member_data['walking'])
        member_group.create_dataset('jumping', data=member_data['jumping'])

    # Create a sub group for the dataset
    dataset_group = f.create_group('dataset')
    meta_data_group = f.create_group('meta_data')


    dataset_group.create_dataset('train',data=[s.values for s in train_segments])
    dataset_group.create_dataset('test',data=[s.values for s in test_segments])


