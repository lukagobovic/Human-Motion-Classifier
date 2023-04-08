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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression





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

    # Append a column of zeros to the combined X, Y, and Z data
    zeros = np.zeros((xyz_data.mean().shape[0], 1))
    features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
                               xyz_data.min(), xyz_data.median(), xyz_data.var(),  xyz_data.kurt(), xyz_data.skew(), zeros))

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

    # Append a column of zeros to the combined X, Y, and Z data
    ones = np.ones((xyz_data.mean().shape[0], 1))
    features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
                               xyz_data.min(), xyz_data.median(), xyz_data.var(), xyz_data.kurt(), xyz_data.skew(), ones))

    np.random.shuffle(features)
    datFrame = pd.DataFrame(features)
    return datFrame

featureData = pd.DataFrame()
for i in range(0,3):
    tempData = pd.DataFrame()
    for j in range(0, len(normalizedData[i]) - 500, 500):
        df = extract_features_walking(normalizedData[i].iloc[j:j+500-1, :],10)
        tempData = pd.concat([tempData,df])
    # print(tempData)

featureData = pd.concat([featureData,tempData])

tempData = pd.DataFrame()
for j in range(0, len(normalizedData[i]) - 500, 500):
    df = extract_features_jumping(normalizedData[3].iloc[j:j+500-1, :],10)
    tempData = pd.concat([tempData,df])
featureData = pd.concat([featureData,tempData])
featureData.interpolate(method = 'linear',inplace=True)

X = featureData.iloc[:,0:24]
y = featureData.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# scaler = StandardScaler()

# l_reg = LogisticRegression(max_iter=10000)
# clf = make_pipeline(StandardScaler(),l_reg)

# clf.fit(X_train,y_train)

y_pred = dtc.predict(X_test)
# y_clf_prob = clf.predict_proba(X_test)
# print('y_pred is:',y_pred)
# print('y_clf_prob is:',y_clf_prob)

acc = accuracy_score(y_test,y_pred)
print('accuracy is:',acc)


# BennettWalkingData = pd.read_csv('MemberData/BennettRawDataWalking.csv')

# CJWalkingData = pd.concat(
#     [pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv'),pd.read_csv('MemberData/CJRawDataJacketWalking.csv'),
#      pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv')]
# )

# df7 = pd.read_csv('MemberData/LukaRawDataJumping.csv')

# LukaJumpingData = df7

# # BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')

# df8 = pd.read_csv('MemberData/CJRawDataJumping.csv')

# CJJumpingData = df8

# all_data = {
#     'Luka': {'walking': LukaWalkingData, 'jumping': LukaJumpingData},
#     # 'Bennett': {'walking': BennettWalkingData, 'jumping': BennettJumpingData},
#     'CJ': {'walking': CJWalkingData, 'jumping': CJJumpingData}
# }



# combinedWalkingData = pd.concat([LukaWalkingData,  CJWalkingData], ignore_index=True) # add BennettWalkingData
# combinedJumpingData = pd.concat([LukaJumpingData,  CJJumpingData], ignore_index=True) # add BennettJumpingData later
# combined_data = pd.concat([combinedWalkingData, combinedJumpingData], ignore_index=True)

# #Concatenate walking and jumping data
# combinedWalkingData = combinedWalkingData.to_numpy()
# np.random.shuffle(combinedWalkingData)
# walking_combined_df = pd.DataFrame(combinedWalkingData)

# combinedJumpingData = combinedJumpingData.to_numpy()
# np.random.shuffle(combinedJumpingData)
# walking_combined_df = pd.DataFrame(combinedJumpingData)

# combined_data = combined_data.to_numpy()
# np.random.shuffle(combined_data)
# test_combined_df = pd.DataFrame(combined_data)

# with h5py.File('./ProjectFile.hdf5', 'w') as f:
#     # Create sub groups for each member
#     for member_name, member_data in all_data.items():
#         member_group = f.create_group(member_name)
#         member_group.create_dataset('walking', data=member_data['walking'])
#         member_group.create_dataset('jumping', data=member_data['jumping'])

#     # Create a sub group for the dataset
#     dataset_group = f.create_group('dataset')

#     # Segment and shuffle all accelerometer data
#     all_segments = []
#     for member_name, member_data in all_data.items():
#         for activity in ['walking', 'jumping']:
#             data = member_data[activity]

#             # Segment the data into 5-second windows
#             num_segments = (len(data) - 500) // 100 + 1
#             segments = [data[(i * 100):(i * 100 + 500)] for i in range(num_segments)]

#             # Label the segments with the member name and activity
#             labels = [f'{member_name}_{activity}' for _ in range(num_segments)]

#             # Combine the segments and labels for this position/activity combination
#             all_segments.extend(list(zip(segments, labels)))

#     # Shuffle the segmented data
#     np.random.shuffle(all_segments)

#     # Split the segmented data into 90% train and 10% test
#     num_train = int(0.9 * len(all_segments))
#     train_segments = all_segments[:num_train]
#     test_segments = all_segments[num_train:]

#     # Create sub groups for train and test datasets
#     train_group = dataset_group.create_group('train')
#     test_group = dataset_group.create_group('test')

#     # Add walking and jumping datasets to train and test sub-groups
#     train_group.create_dataset('walking', data=[seg[0] for seg in train_segments if 'walking' in seg[1]])
#     train_group.create_dataset('jumping', data=[seg[0] for seg in train_segments if 'jumping' in seg[1]])
#     test_group.create_dataset('walking', data=[seg[0] for seg in test_segments if 'walking' in seg[1]])
#     test_group.create_dataset('jumping', data=[seg[0] for seg in test_segments if 'jumping' in seg[1]])


# # # Load data from HDF5 file
# # with h5py.File('combined_data.hdf5', 'r') as f:
# #     train_X = f['train_X'][:]
# #     train_y = f['train_y'][:]
# #     test_X = f['test_X'][:]
# #     test_y = f['test_y'][:]

# # # Apply moving average filter
# # # train_X = savgol_filter(train_X, window_length=5, polyorder=3, axis=1)
# # # test_X = savgol_filter(test_X, window_length=5, polyorder=3, axis=1)

# # # # Detect and remove outliers
# # # train_X = train_X[~((train_X - np.mean(train_X, axis=1, keepdims=True)) > 2 * np.std(train_X, axis=1, keepdims=True))]
# # # train_y = train_y[~((train_X - np.mean(train_X, axis=1, keepdims=True)) > 2 * np.std(train_X, axis=1, keepdims=True))]
# # # test_X = test_X[~((test_X - np.mean(test_X, axis=1, keepdims=True)) > 2 * np.std(test_X, axis=1, keepdims=True))]
# # # test_y = test_y[~((test_X - np.mean(test_X, axis=1, keepdims=True)) > 2 * np.std(test_X, axis=1, keepdims=True))]

# # # # Normalize the data
# # # train_X = (train_X - np.mean(train_X)) / np.std(train_X)
# # # test_X = (test_X - np.mean(test_X)) / np.std(test_X)

# # # Define a function to extract features
# # # def extract_features(data):
# # #     features = []
# # #     for i in range(data.shape[0]):
# # #         window = data[i, :]
# # #         feature = []
# # #         feature.append(np.max(window[:, 0]))  # Maximum acceleration in x-direction
# # #         feature.append(np.max(window[:, 1]))  # Maximum acceleration in y-direction
# # #         feature.append(np.max(window[:, 2]))  # Maximum acceleration in z-direction
# # #         feature.append(np.min(window[:, 0]))  # Minimum acceleration in x-direction
# # #         feature.append(np.min(window[:, 1]))  # Minimum acceleration in y-direction
# # #         feature.append(np.min(window[:, 2]))  # Minimum acceleration in z-direction
# # #         feature.append(np.mean(window[:, 0]))  # Mean acceleration in x-direction
# # #         feature.append(np.mean(window[:, 1]))  # Mean acceleration in y-direction
# # #         feature.append(np.mean(window[:, 2]))  # Mean acceleration in z-direction
# # #         feature.append(np.std(window[:, 0]))  # Standard deviation of acceleration in x-direction
# # #         feature.append(np.std(window[:, 1]))  # Standard deviation of acceleration in y-direction
# # #         feature.append(np.std(window[:, 2]))  # Standard deviation of acceleration in z-direction
# # #         features.append(feature)
# # #     return np.array(features)

# # def extract_features(data):
# #     # Create an empty array to hold the feature vectors
# #     features = np.zeros((data.shape[0], 12))

# #     # Iterate over each data row and extract the features
# #     for i in range(data.shape[0]):
# #         # Extract the data from the row
# #         row_data = data[i, 1:4]  # extract data from columns 2, 3, and 4

# #         # Compute the features
# #         max_x = np.max(row_data[0])
# #         max_y = np.max(row_data[1])
# #         max_z = np.max(row_data[2])
# #         min_x = np.min(row_data[0])
# #         min_y = np.min(row_data[1])
# #         min_z = np.min(row_data[2])
# #         mean_x = np.mean(row_data[0])
# #         mean_y = np.mean(row_data[1])
# #         mean_z = np.mean(row_data[2])
# #         std_x = np.std(row_data[0])
# #         std_y = np.std(row_data[1])
# #         std_z = np.std(row_data[2])

# #         # Store the features in the features array
# #         features[i, :] = (max_x, max_y, max_z, min_x, min_y, min_z, mean_x, mean_y, mean_z, std_x, std_y, std_z)

# #     # Create a column of labels
# #     labels = np.ones((data.shape[0], 1))

# #     # Add the labels column to the feature array
# #     all_features = np.hstack((features, labels))

# #     return all_features

# # # Extract features from training and testing sets
# # train_features = extract_features(train_X[:, 1:]) # exclude the first column (time)
# # test_features = extract_features(test_X[:, 1:]) # exclude the first column (time)
# # training_labels = np.concatenate((np.zeros((walking_filtered.shape[0], 1)),
# #                                   np.ones((jumping_filtered.shape[0], 1))), axis=0)
# # test_labels = np.concatenate((np.zeros((test_walking_features.shape[0], 1)),
# #                               np.ones((test_jumping_features.shape[0], 1))), axis=0)


# # print(train_features)
# # print(test_features)

# # # Train a logistic regression model
# # l_reg = LogisticRegression(max_iter=10000)
# # scaler = StandardScaler()
# # clf = make_pipeline(scaler,l_reg)
# # clf.fit(train_features, train_y)

# # # # Predict labels for test set
# # y_pred = clf.predict(test_features)

# # # # Calculate accuracy score
# # accuracy = accuracy_score(test_y, y_pred)

# # print(f"Test accuracy: {accuracy:.2f}")