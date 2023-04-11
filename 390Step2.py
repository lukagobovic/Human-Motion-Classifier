import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, f1_score
from matplotlib import pyplot as plt
import h5py


df1 = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv',nrows = 30000)
df2 = pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv',nrows = 30000)
df3 = pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv',nrows = 30000)

df4 = pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv',nrows = 18000)
df5 = pd.read_csv('MemberData/CJRawDataJacketWalking.csv',nrows = 18000)
df6 = pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv',nrows = 18000)

df7 = pd.read_csv('MemberData/BennettRawDataBackPocketWalking.csv',nrows = 27000)
df8 = pd.read_csv('MemberData/BennettRawDataFrontPocketWalking.csv',nrows = 27000)
df9 = pd.read_csv('MemberData/BennettRawDataJacketWalking.csv',nrows = 27000)

listOfWalkingData = [df1,df2,df3,df4,df5,df6,df7,df8,df9]

for df in listOfWalkingData:
  df = df.iloc[:,1:]
  q1 = df.quantile(0.25)
  q3 = df.quantile(0.75)
  iqr = q3 - q1
  df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))] = np.nan

  df.interpolate(method='linear', inplace=True)
  
  data = df.rolling(5).mean().dropna()
    # Remove outliers
    # Normalize the data
  scaler = MinMaxScaler()
  data.iloc[:,1:-2] = scaler.fit_transform(data.iloc[:,1:-2])

listOfWalkingDataDF = pd.DataFrame()
for df in listOfWalkingData:
  listOfWalkingDataDF = pd.concat([listOfWalkingDataDF,df])

# print(listOfWalkingDataDF)
listOfWalkingDataDF['activity'] = 0  

listOfJumpingData = pd.DataFrame()
df10 = pd.read_csv('MemberData/LukaRawDataJumping.csv',nrows = 30000)
df11 = pd.read_csv('MemberData/BennettRawDataJumping.csv')
df12 = pd.read_csv('MemberData/CJRawDataJumping.csv',nrows = 16500)
listOfJumpingData = [df10,df11,df12]

for df in listOfJumpingData:
  df = df.iloc[:,1:]
  q1 = df.quantile(0.25)
  q3 = df.quantile(0.75)
  iqr = q3 - q1
  df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))] = np.nan

  df.interpolate(method='linear', inplace=True)
  
  data = df.rolling(5).mean().dropna()
    # Remove outliers
    # Normalize the data
  scaler = MinMaxScaler()
  data.iloc[:,1:-2] = scaler.fit_transform(data.iloc[:,1:-2])

listOfJumpingDataDF = pd.DataFrame()
for df in listOfJumpingData:
  listOfJumpingDataDF = pd.concat([listOfJumpingDataDF,df])

# print(listOfWalkingDataDF)
listOfJumpingDataDF['activity'] = 1 

LukaWalkingData = pd.concat(
    [df1,df2,df3]
)
CJWalkingData = pd.concat(
     [df4,df5,df6]
 )

BennettWalkingData = pd.concat(
   [df7,df8,df9]
)

all_data = {
    'Luka': {'walking': LukaWalkingData, 'jumping': df10},
    'Bennett': {'walking': BennettWalkingData,'jumping': df11},
    'CJ': {'walking': CJWalkingData, 'jumping': df12}
}


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
