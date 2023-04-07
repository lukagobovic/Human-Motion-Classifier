import pandas as pd
import numpy as np
import h5py
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew



df1 = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv',nrows = 30005)
df2 = pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv',nrows = 30005)
df3 = pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv',nrows = 30005)
df4 = pd.read_csv('MemberData/LukaRawDataJumping.csv')

# df5 = pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv',nrows = 30005)
# df6 = pd.read_csv('MemberData/CJRawDataJacketWalking.csv',nrows = 30005)
# df7 = pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv',nrows = 30005)
listOfData = []
listOfData.append(df1)
listOfData.append(df2)
listOfData.append(df3)
listOfData.append(df4)
# listOfData.append(df5)
# listOfData.append(df6)

#df7 = pd.read_csv('MemberData/BennettRawDataWalking.csv')

LukaWalkingData = pd.concat(
    [df1,df2,df3]
)
# CJWalkingData = pd.concat(
#     [df4,df5,df6]
# )

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
for i in range(0,4):
    normalizedData.append(normalizeData(listOfData[i],10))

# def compute_stats(data):
#   return [np.mean(data), np.std(data), np.min(data), np.max(data), np.median(data), np.var(data), skew(data),np.sqrt(np.mean(data**2))]

# def extract_features(data,wsize):
#     data = data.iloc[:,1:]
#     # Create empty list to store features for each window
#     features = [[]]
#     # Iterate over each window of data
#     for i in range(0, len(data) - wsize, wsize):
#         windowFeatures = []
#         # Extract the x, y, z accelerometer data for the current window
#         window_data = data.iloc[i:i+wsize-1, :]
#         x_data = window_data.iloc[:, 0]
#         y_data = window_data.iloc[:, 1]
#         z_data = window_data.iloc[:, 2]
#         # Compute summary statistics for each axis over the window
#         x_stats = compute_stats(x_data)
#         y_stats = compute_stats(y_data)
#         z_stats = compute_stats(z_data)
#         # Append the summary statistics to the features list
#         windowFeatures.append(x_stats)
#         windowFeatures.append(y_stats)
#         windowFeatures.append(z_stats)
#         features.insert(i,windowFeatures)
#     # Convert the features list to a numpy array and return it
#     return features

# def extract_features(windows):
#     # Create an empty array to hold the feature vectors
#     features = np.zeros((windows.shape[0], 10, 4))

#     # Iterate over each time window and extract the features
#     for i in range(windows.shape[2]):
#         for j in range(windows.shape[0]):
#             # Extract the data from the window
#             window_data = windows[j, :, i]

#             # Compute the features
#             max_val = np.max(window_data)
#             min_val = np.min(window_data)
#             range_val = max_val - min_val
#             mean_val = np.mean(window_data)
#             median_val = np.median(window_data)
#             var_val = np.var(window_data)
#             skew_val = skew(window_data)
#             rms_val = np.sqrt(np.mean(window_data ** 2))
#             kurt_val = np.mean((window_data - np.mean(window_data)) ** 4) / (np.var(window_data) ** 2)
#             std_val = np.std(window_data)

#             # Store the features in the features array
#             features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val,
#                                  rms_val, kurt_val, std_val)

#     return features


def extract_features(data,wsize):
    featuresX = []
    featuresY = []
    featuresZ = []
    features = []
    featuresX.append(data.iloc[:,1].rolling(window=wsize).mean())
    featuresY.append(data.iloc[:,2].rolling(window=wsize).mean())
    featuresZ.append(data.iloc[:,3].rolling(window=wsize).mean())
    featuresX.append(data.iloc[:,1].rolling(window=wsize).std())
    featuresY.append(data.iloc[:,2].rolling(window=wsize).std())
    featuresZ.append(data.iloc[:,3].rolling(window=wsize).std())
    featuresX.append(data.iloc[:,1].rolling(window=wsize).max())
    featuresY.append(data.iloc[:,2].rolling(window=wsize).max())
    featuresZ.append(data.iloc[:,3].rolling(window=wsize).max())
    featuresX.append(data.iloc[:,1].rolling(window=wsize).min())
    featuresY.append(data.iloc[:,2].rolling(window=wsize).min())
    featuresZ.append(data.iloc[:,3].rolling(window=wsize).min())
    featuresX.append(data.iloc[:,1].rolling(window=wsize).kurt())
    featuresY.append(data.iloc[:,2].rolling(window=wsize).kurt())
    featuresZ.append(data.iloc[:,3].rolling(window=wsize).kurt())
    featuresX.append(data.iloc[:,1].rolling(window=wsize).skew())
    featuresY.append(data.iloc[:,2].rolling(window=wsize).skew())
    featuresZ.append(data.iloc[:,3].rolling(window=wsize).skew())

    features.append(featuresX)
    features.append(featuresY)
    features.append(featuresZ)
    np.random.shuffle(features)
    return np.array(features)


featureData = []
for i in range(0,4):
    for j in range(0, len(normalizedData[i]) - 500, 500):
        featureData.append(extract_features(normalizedData[i].iloc[j:j+500-1, :],10))
        print(featureData[i].shape)

# zeros_col = np.zeros((featureData[0][0].shape[0],1))
# testArray = np.hstack((featureData[0][0],zeros_col))
# print(testArray)



# BennettWalkingData = pd.read_csv('MemberData/BennettRawDataWalking.csv')

# CJWalkingData = pd.concat(
#     [pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv'),pd.read_csv('MemberData/CJRawDataJacketWalking.csv'),
#      pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv')]
# )

df7 = pd.read_csv('MemberData/LukaRawDataJumping.csv')

LukaJumpingData = df7

# BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')

df8 = pd.read_csv('MemberData/CJRawDataJumping.csv')

CJJumpingData = df8

all_data = {
    'Luka': {'walking': LukaWalkingData, 'jumping': LukaJumpingData},
    # 'Bennett': {'walking': BennettWalkingData, 'jumping': BennettJumpingData},
    'CJ': {'walking': CJWalkingData, 'jumping': CJJumpingData}
}



combinedWalkingData = pd.concat([LukaWalkingData,  CJWalkingData], ignore_index=True) # add BennettWalkingData
combinedJumpingData = pd.concat([LukaJumpingData,  CJJumpingData], ignore_index=True) # add BennettJumpingData later
combined_data = pd.concat([combinedWalkingData, combinedJumpingData], ignore_index=True)

#Concatenate walking and jumping data
combinedWalkingData = combinedWalkingData.to_numpy()
np.random.shuffle(combinedWalkingData)
walking_combined_df = pd.DataFrame(combinedWalkingData)

combinedJumpingData = combinedJumpingData.to_numpy()
np.random.shuffle(combinedJumpingData)
walking_combined_df = pd.DataFrame(combinedJumpingData)

combined_data = combined_data.to_numpy()
np.random.shuffle(combined_data)
test_combined_df = pd.DataFrame(combined_data)

with h5py.File('./ProjectFile.hdf5', 'w') as f:
    # Create sub groups for each member
    for member_name, member_data in all_data.items():
        member_group = f.create_group(member_name)
        member_group.create_dataset('walking', data=member_data['walking'])
        member_group.create_dataset('jumping', data=member_data['jumping'])

    # Create a sub group for the dataset
    dataset_group = f.create_group('dataset')

    # Segment and shuffle all accelerometer data
    all_segments = []
    for member_name, member_data in all_data.items():
        for activity in ['walking', 'jumping']:
            data = member_data[activity]

            # Segment the data into 5-second windows
            num_segments = (len(data) - 500) // 100 + 1
            segments = [data[(i * 100):(i * 100 + 500)] for i in range(num_segments)]

            # Label the segments with the member name and activity
            labels = [f'{member_name}_{activity}' for _ in range(num_segments)]

            # Combine the segments and labels for this position/activity combination
            all_segments.extend(list(zip(segments, labels)))

    # Shuffle the segmented data
    np.random.shuffle(all_segments)

    # Split the segmented data into 90% train and 10% test
    num_train = int(0.9 * len(all_segments))
    train_segments = all_segments[:num_train]
    test_segments = all_segments[num_train:]

    # Create sub groups for train and test datasets
    train_group = dataset_group.create_group('train')
    test_group = dataset_group.create_group('test')

    # Add walking and jumping datasets to train and test sub-groups
    train_group.create_dataset('walking', data=[seg[0] for seg in train_segments if 'walking' in seg[1]])
    train_group.create_dataset('jumping', data=[seg[0] for seg in train_segments if 'jumping' in seg[1]])
    test_group.create_dataset('walking', data=[seg[0] for seg in test_segments if 'walking' in seg[1]])
    test_group.create_dataset('jumping', data=[seg[0] for seg in test_segments if 'jumping' in seg[1]])


# # Load data from HDF5 file
# with h5py.File('combined_data.hdf5', 'r') as f:
#     train_X = f['train_X'][:]
#     train_y = f['train_y'][:]
#     test_X = f['test_X'][:]
#     test_y = f['test_y'][:]

# # Apply moving average filter
# # train_X = savgol_filter(train_X, window_length=5, polyorder=3, axis=1)
# # test_X = savgol_filter(test_X, window_length=5, polyorder=3, axis=1)

# # # Detect and remove outliers
# # train_X = train_X[~((train_X - np.mean(train_X, axis=1, keepdims=True)) > 2 * np.std(train_X, axis=1, keepdims=True))]
# # train_y = train_y[~((train_X - np.mean(train_X, axis=1, keepdims=True)) > 2 * np.std(train_X, axis=1, keepdims=True))]
# # test_X = test_X[~((test_X - np.mean(test_X, axis=1, keepdims=True)) > 2 * np.std(test_X, axis=1, keepdims=True))]
# # test_y = test_y[~((test_X - np.mean(test_X, axis=1, keepdims=True)) > 2 * np.std(test_X, axis=1, keepdims=True))]

# # # Normalize the data
# # train_X = (train_X - np.mean(train_X)) / np.std(train_X)
# # test_X = (test_X - np.mean(test_X)) / np.std(test_X)

# # Define a function to extract features
# # def extract_features(data):
# #     features = []
# #     for i in range(data.shape[0]):
# #         window = data[i, :]
# #         feature = []
# #         feature.append(np.max(window[:, 0]))  # Maximum acceleration in x-direction
# #         feature.append(np.max(window[:, 1]))  # Maximum acceleration in y-direction
# #         feature.append(np.max(window[:, 2]))  # Maximum acceleration in z-direction
# #         feature.append(np.min(window[:, 0]))  # Minimum acceleration in x-direction
# #         feature.append(np.min(window[:, 1]))  # Minimum acceleration in y-direction
# #         feature.append(np.min(window[:, 2]))  # Minimum acceleration in z-direction
# #         feature.append(np.mean(window[:, 0]))  # Mean acceleration in x-direction
# #         feature.append(np.mean(window[:, 1]))  # Mean acceleration in y-direction
# #         feature.append(np.mean(window[:, 2]))  # Mean acceleration in z-direction
# #         feature.append(np.std(window[:, 0]))  # Standard deviation of acceleration in x-direction
# #         feature.append(np.std(window[:, 1]))  # Standard deviation of acceleration in y-direction
# #         feature.append(np.std(window[:, 2]))  # Standard deviation of acceleration in z-direction
# #         features.append(feature)
# #     return np.array(features)

# def extract_features(data):
#     # Create an empty array to hold the feature vectors
#     features = np.zeros((data.shape[0], 12))

#     # Iterate over each data row and extract the features
#     for i in range(data.shape[0]):
#         # Extract the data from the row
#         row_data = data[i, 1:4]  # extract data from columns 2, 3, and 4

#         # Compute the features
#         max_x = np.max(row_data[0])
#         max_y = np.max(row_data[1])
#         max_z = np.max(row_data[2])
#         min_x = np.min(row_data[0])
#         min_y = np.min(row_data[1])
#         min_z = np.min(row_data[2])
#         mean_x = np.mean(row_data[0])
#         mean_y = np.mean(row_data[1])
#         mean_z = np.mean(row_data[2])
#         std_x = np.std(row_data[0])
#         std_y = np.std(row_data[1])
#         std_z = np.std(row_data[2])

#         # Store the features in the features array
#         features[i, :] = (max_x, max_y, max_z, min_x, min_y, min_z, mean_x, mean_y, mean_z, std_x, std_y, std_z)

#     # Create a column of labels
#     labels = np.ones((data.shape[0], 1))

#     # Add the labels column to the feature array
#     all_features = np.hstack((features, labels))

#     return all_features

# # Extract features from training and testing sets
# train_features = extract_features(train_X[:, 1:]) # exclude the first column (time)
# test_features = extract_features(test_X[:, 1:]) # exclude the first column (time)
# training_labels = np.concatenate((np.zeros((walking_filtered.shape[0], 1)),
#                                   np.ones((jumping_filtered.shape[0], 1))), axis=0)
# test_labels = np.concatenate((np.zeros((test_walking_features.shape[0], 1)),
#                               np.ones((test_jumping_features.shape[0], 1))), axis=0)


# print(train_features)
# print(test_features)

# # Train a logistic regression model
# l_reg = LogisticRegression(max_iter=10000)
# scaler = StandardScaler()
# clf = make_pipeline(scaler,l_reg)
# clf.fit(train_features, train_y)

# # # Predict labels for test set
# y_pred = clf.predict(test_features)

# # # Calculate accuracy score
# accuracy = accuracy_score(test_y, y_pred)

# print(f"Test accuracy: {accuracy:.2f}")