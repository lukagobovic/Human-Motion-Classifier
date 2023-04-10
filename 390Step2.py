import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from scipy.signal import savgol_filter
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
listOfWalkingData = pd.DataFrame()
listOfWalkingData = pd.concat([df1,df2,df3,df4,df5,df6])
listOfWalkingData['activity'] = 0

listOfJumpingData = pd.DataFrame()
df7 = pd.read_csv('MemberData/LukaRawDataJumping.csv',nrows = 30000)
# BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')
df8 = pd.read_csv('MemberData/CJRawDataJumping.csv',nrows = 16500)

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


listOfJumpingData = pd.concat([df7,df8])
listOfJumpingData['activity'] = 1

lisfOfCombinedData = pd.concat([listOfWalkingData,listOfJumpingData])
data = lisfOfCombinedData

# Define window size 
window_size = 5

# Segment the data into windows of a given size
def segment_data(data, window_size):
  num_samples = int(np.floor(data.shape[0] / (window_size * 100)))  # 50 Hz sampling rate
  segments = []
  for i in range(num_samples):
    start_idx = i * window_size * 100
    end_idx = start_idx + window_size * 100
    segment = data.iloc[start_idx:end_idx]
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

    dataset_group.create_dataset('train',data=[s.values for s in train_segments])
    dataset_group.create_dataset('test',data=[s.values for s in test_segments])