from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

def preprocess_data(data):
  data = data.rolling(window_size).mean().dropna()
  # Remove outliers
  data = data[(np.abs(data.x - data.x.mean()) / data.x.std()) < 3]
  data = data[(np.abs(data.y - data.y.mean()) / data.y.std()) < 3]
  data = data[(np.abs(data.z - data.z.mean()) / data.z.std()) < 3]

  # Normalize the data
  data = (data - data.mean()) / data.std()

  # Extract features
  features = [
    np.max(data.x),
    np.min(data.x),
    np.ptp(data.x),
    np.mean(data.x),
    np.median(data.x),
    np.var(data.x),
    np.std(data.x),
    np.max(data.y),
    np.min(data.y),
    np.ptp(data.y),
    np.mean(data.y),
    np.median(data.y),
    np.var(data.y),
    np.std(data.y),
    np.max(data.z),
    np.min(data.z),
    np.ptp(data.z),
    np.mean(data.z),
    np.median(data.z),
    np.var(data.z),
    np.std(data.z),
    np.max(data.total_acceleration),
    np.min(data.total_acceleration),
    np.ptp(data.total_acceleration),
    np.mean(data.total_acceleration),
    np.median(data.total_acceleration),
    np.var(data.total_acceleration),
    np.std(data.total_acceleration),
  ]

  return features

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

fileName = 'LukaRawDataFrontPocketWalking'

# Load the feature data
originalData = pd.read_csv('MemberData/'+fileName+'.csv',nrows = 18000)
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
segments = segment_data(originalData, window_size)
np.random.shuffle(segments)

train_features = [preprocess_data(segment) for segment in segments]

# X_train = pd.DataFrame(X_train)
y_pred = model.predict(train_features)

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