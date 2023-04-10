from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

def preprocess_data(data):
  data = data.rolling(5).mean().dropna()

  # Remove outliers
  data = data[(np.abs(data.x - data.x.mean()) / data.x.std()) < 2.5]
  data = data[(np.abs(data.y - data.y.mean()) / data.y.std()) < 2.5]
  data = data[(np.abs(data.z - data.z.mean()) / data.z.std()) < 2.5]

  # Normalize the data
  data = (data - data.mean()) / data.std()
  data = data[:450]

  # Extract features
  # Extract features for each window of 500 rows
  window_size = 10
  num_windows = len(data) // window_size
  features = []
  for i in range(num_windows):
    window_data = data.iloc[i*window_size:(i+1)*window_size]
    window_features = [
        np.max(window_data.x),
        np.min(window_data.x),
        np.ptp(window_data.x),
        np.mean(window_data.x),
        np.median(window_data.x),
        np.var(window_data.x),
        np.std(window_data.x),
        np.max(window_data.y),
        np.min(window_data.y),
        np.ptp(window_data.y),
        np.mean(window_data.y),
        np.median(window_data.y),
        np.var(window_data.y),
        np.std(window_data.y),
        np.max(window_data.z),
        np.min(window_data.z),
        np.ptp(window_data.z),
        np.mean(window_data.z),
        np.median(window_data.z),
        np.var(window_data.z),
        np.std(window_data.z),
        np.max(window_data.total_acceleration),
        np.min(window_data.total_acceleration),
        np.ptp(window_data.total_acceleration),
        np.mean(window_data.total_acceleration),
        np.median(window_data.total_acceleration),
        np.var(window_data.total_acceleration),
        np.std(window_data.total_acceleration),
    ]
    features.append(window_features)

  return features

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

fileName = 'LukaRawDataJumping'

# Load the feature data
originalData = pd.read_csv('MemberData/'+fileName+'.csv')
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

train_features = np.concatenate([preprocess_data(segment) for segment in segments])

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