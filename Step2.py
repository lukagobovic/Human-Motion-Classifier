import pandas as pd
import numpy as np
import h5py
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



LukaWalkingData = pd.concat(
    [pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv'),pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv'),
     pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv')]
)

BennettWalkingData = pd.read_csv('MemberData/BennettRawDataWalking.csv')

CJWalkingData = pd.concat(
    [pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv'),pd.read_csv('MemberData/CJRawDataJacketWalking.csv'),
     pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv')]
)

LukaJumpingData = pd.read_csv('MemberData/LukaRawDataJumping.csv')

BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')

CJJumpingData = pd.read_csv('MemberData/CJRawDataJumping.csv')


all_data = {
    'Luka': {'walking': LukaWalkingData, 'jumping': LukaJumpingData},
    'Bennett': {'walking': BennettWalkingData, 'jumping': BennettJumpingData},
    'CJ': {'walking': CJWalkingData, 'jumping': CJJumpingData}
}

# Concatenate walking and jumping data
combinedWalkingData = pd.concat([LukaWalkingData, BennettWalkingData, CJWalkingData], ignore_index=True)
combinedJumpingData = pd.concat([LukaJumpingData, BennettJumpingData, CJJumpingData], ignore_index=True)

combined_data = pd.concat([combinedWalkingData, combinedJumpingData], ignore_index=True)

window_size = 5 * 100  # 5 seconds at 100 Hz sampling rate
num_windows = len(combined_data) // window_size
segmented_data = [combined_data[i*window_size : (i+1)*window_size] for i in range(num_windows)]

np.random.shuffle(segmented_data)

shuffled_data = pd.concat(segmented_data, ignore_index=True)

train_data, test_data = train_test_split(shuffled_data, test_size=0.1, random_state=42)
print(train_data)


# train_indices = window_indices[:int(0.9 * num_windows)]
# test_indices = window_indices[int(0.9 * num_windows):]
# print(train_indices)

# train_data_start = train_indices[0] * window_size
# train_data_end = (train_indices[-1] + 1) * window_size
# train_data = combined_data.iloc[train_data_start:train_data_end, :]

# test_data_start = test_indices[0] * window_size
# test_data_end = (test_indices[-1] + 1) * window_size
# test_data = combined_data.iloc[test_data_start:test_data_end, :]
# print(test_data)



# # Save data to HDF5 file
with h5py.File('combined_data.hdf5', 'w') as f:
    # Store training data
    f.create_dataset('train_X', data=train_data.values)
    f.create_dataset('train_y', data=np.zeros((len(train_data), 1)))

    # Store test data
    f.create_dataset('test_X', data=test_data.values)
    f.create_dataset('test_y', data=np.zeros((len(test_data), 1)))

# Load data from HDF5 file
with h5py.File('combined_data.hdf5', 'r') as f:
    train_X = f['train_X'][:]
    train_y = f['train_y'][:]
    test_X = f['test_X'][:]
    test_y = f['test_y'][:]

# Apply moving average filter
# train_X = savgol_filter(train_X, window_length=5, polyorder=3, axis=1)
# test_X = savgol_filter(test_X, window_length=5, polyorder=3, axis=1)

# # Detect and remove outliers
# train_X = train_X[~((train_X - np.mean(train_X, axis=1, keepdims=True)) > 2 * np.std(train_X, axis=1, keepdims=True))]
# train_y = train_y[~((train_X - np.mean(train_X, axis=1, keepdims=True)) > 2 * np.std(train_X, axis=1, keepdims=True))]
# test_X = test_X[~((test_X - np.mean(test_X, axis=1, keepdims=True)) > 2 * np.std(test_X, axis=1, keepdims=True))]
# test_y = test_y[~((test_X - np.mean(test_X, axis=1, keepdims=True)) > 2 * np.std(test_X, axis=1, keepdims=True))]

# # Normalize the data
# train_X = (train_X - np.mean(train_X)) / np.std(train_X)
# test_X = (test_X - np.mean(test_X)) / np.std(test_X)

# Define a function to extract features
def extract_features(data):
    features = []
    for i in range(data.shape[0]):
        window = data[i, :, :]
        feature = []
        feature.append(np.max(window[:, 0]))  # Maximum acceleration in x-direction
        feature.append(np.max(window[:, 1]))  # Maximum acceleration in y-direction
        feature.append(np.max(window[:, 2]))  # Maximum acceleration in z-direction
        feature.append(np.min(window[:, 0]))  # Minimum acceleration in x-direction
        feature.append(np.min(window[:, 1]))  # Minimum acceleration in y-direction
        feature.append(np.min(window[:, 2]))  # Minimum acceleration in z-direction
        feature.append(np.mean(window[:, 0]))  # Mean acceleration in x-direction
        feature.append(np.mean(window[:, 1]))  # Mean acceleration in y-direction
        feature.append(np.mean(window[:, 2]))  # Mean acceleration in z-direction
        feature.append(np.std(window[:, 0]))  # Standard deviation of acceleration in x-direction
        feature.append(np.std(window[:, 1]))  # Standard deviation of acceleration in y-direction
        feature.append(np.std(window[:, 2]))  # Standard deviation of acceleration in z-direction
        features.append(feature)
    return np.array(features)

# Extract features from training and testing sets
train_features = extract_features(train_X)
test_features = extract_features(test_X)

# Train a logistic regression model
# clf = LogisticRegression(random_state=42).fit(train_features, train_y)

# # Predict labels for test set
# y_pred = clf.predict(test_features)

# # Calculate accuracy score
# accuracy = accuracy_score(test_y, y_pred)

# print(f"Test accuracy: {accuracy:.2f}")