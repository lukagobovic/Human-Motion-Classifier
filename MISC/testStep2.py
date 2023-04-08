import pandas as pd
import numpy as np
import h5py
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



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

# Divide data into 5-second windows
window_size = 5 * 100  # 5 seconds at 50 Hz sampling rate
num_windows = len(combined_data) // window_size

# Shuffle segmented data and split into train and test sets
np.random.seed(42)
window_indices = np.arange(num_windows)
np.random.shuffle(window_indices)

n_samples, n_features = combined_data.shape

# Split the data into training and testing sets
n_train = int(0.9 * len(combined_data) // window_size)
n_test = len(combined_data) // window_size - n_train
train_indices = np.arange(n_train)
test_indices = np.arange(n_train, n_train + n_test)

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

# Extract features from the training set
train_features = np.zeros((n_train, n_features))
for i in range(n_train):
    train_data = combined_data.iloc[train_indices[i] * window_size : (train_indices[i] + 1) * window_size]
    train_features[i] = extract_features(train_data)

# Extract features from the testing set
test_features = np.zeros((n_test, n_features))
for i in range(n_test):
    test_data = combined_data.iloc[test_indices[i] * window_size : (test_indices[i] + 1) * window_size]
    test_features[i] = extract_features(test_data)

# # Save data to HDF5 file
# with h5py.File('combined_data.hdf5', 'w') as f:
#     # Store training data
#     f.create_dataset('train_X', data=train_data.values)
#     f.create_dataset('train_y', data=np.zeros((len(train_data), 1)))

#     # Store test data
#     f.create_dataset('test_X', data=test_data.values)
#     f.create_dataset('test_y', data=np.zeros((len(test_data), 1)))

# # Load data from HDF5 file
# with h5py.File('combined_data.h5', 'r') as f:
#     train_X = f['train/X'][:]
#     train_y = f['train/y'][:]
#     test_X = f['test/X'][:]
#     test_y = f['test/y'][:]

with h5py.File('accelerometer_data.h5', 'r') as f:
    X_train = f['train/X'][:]
    y_train = f['train/y'][:]
    X_test = f['test/X'][:]
    y_test = f['test/y'][:]

# Combine all windows into one DataFrame
n_train, window_size, n_features = X_train.shape
n_test = X_test.shape[0]
X_train = X_train.reshape((n_train * window_size, n_features))
X_test = X_test.reshape((n_test * window_size, n_features))
y_train = np.repeat(y_train, window_size)
y_test = np.repeat(y_test, window_size)
combined_data = pd.DataFrame(np.concatenate([X_train, X_test]), columns=['X', 'Y', 'Z'])

# Normalize the data
combined_data = (combined_data - combined_data.mean()) / combined_data.std()

# Define training and testing indices
n_windows = n_train + n_test
window_indices = np.arange(n_windows)
np.random.shuffle(window_indices)
train_indices = window_indices[:n_train]
test_indices = window_indices[n_train:]

# Train the logistic regression model
model = LogisticRegression()
train_accuracy = []
test_accuracy = []
for i in range(n_train):
    train_data = combined_data.iloc[train_indices[i] * window_size : (train_indices[i] + 1) * window_size]
    X_train = train_data.values
    y_train = np.unique(y_train[train_indices[i] * window_size : (train_indices[i] + 1) * window_size])
    model.fit(X_train, y_train)
    train_accuracy.append(model.score(X_train, y_train))
    test_data = combined_data.iloc[test_indices * window_size : (test_indices + 1) * window_size]
    X_test = test_data.values
    y_test = np.unique(y_test[test_indices * window_size : (test_indices + 1) * window_size])
    test_accuracy.append(model.score(X_test, y_test))
    
# Print training and testing accuracies
print("Training accuracy:", np.mean(train_accuracy))
print("Testing accuracy:", np.mean(test_accuracy))

import pandas as pd
import numpy as np
import h5py
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



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

# Divide data into 5-second windows
window_size = 5 * 100  # 5 seconds at 50 Hz sampling rate
num_windows = len(combined_data) // window_size

# Shuffle segmented data and split into train and test sets
np.random.seed(42)
window_indices = np.arange(num_windows)
np.random.shuffle(window_indices)

n_samples, n_features = combined_data.shape

# Split the data into training and testing sets
n_train = int(0.9 * len(combined_data) // window_size)
n_test = len(combined_data) // window_size - n_train
train_indices = np.arange(n_train)
test_indices = np.arange(n_train, n_train + n_test)

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

# Extract features from the training set
train_features = np.zeros((n_train, n_features))
for i in range(n_train):
    train_data = combined_data.iloc[train_indices[i] * window_size : (train_indices[i] + 1) * window_size]
    train_features[i] = extract_features(train_data)

# Extract features from the testing set
test_features = np.zeros((n_test, n_features))
for i in range(n_test):
    test_data = combined_data.iloc[test_indices[i] * window_size : (test_indices[i] + 1) * window_size]
    test_features[i] = extract_features(test_data)

# # Save data to HDF5 file
# with h5py.File('combined_data.hdf5', 'w') as f:
#     # Store training data
#     f.create_dataset('train_X', data=train_data.values)
#     f.create_dataset('train_y', data=np.zeros((len(train_data), 1)))

#     # Store test data
#     f.create_dataset('test_X', data=test_data.values)
#     f.create_dataset('test_y', data=np.zeros((len(test_data), 1)))

# # Load data from HDF5 file
# with h5py.File('combined_data.h5', 'r') as f:
#     train_X = f['train/X'][:]
#     train_y = f['train/y'][:]
#     test_X = f['test/X'][:]
#     test_y = f['test/y'][:]

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

# Extract features from training and testing sets
# train_features = extract_features(train_X)
# test_features = extract_features(test_X)

# # Train a logistic regression model
# clf = LogisticRegression(random_state=42).fit(train_features, train_y)

# # Predict labels for test set
# y_pred = clf.predict(test_features)

# # Calculate accuracy score
# accuracy = accuracy_score(test_y, y_pred)

# print(f"Test accuracy: {accuracy:.2f}")

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train a logistic regression model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Evaluate the model on the testing set
accuracy = model.score(test_features, test_labels)
print('Test accuracy:', accuracy)