from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import skew
from sklearn.metrics import recall_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


import pickle





df1 = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv',nrows = 30005)
df2 = pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv',nrows = 30005)
df3 = pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv',nrows = 30005)

df4 = pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv',nrows = 30005)
df5 = pd.read_csv('MemberData/CJRawDataJacketWalking.csv',nrows = 30005)
df6 = pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv',nrows = 30005)
listOfWalkingData = pd.DataFrame()
listOfWalkingData = pd.concat([df1,df2,df3,df4,df5,df6])
listOfWalkingData['labels'] = 0



listOfJumpingData = pd.DataFrame()
df7 = pd.read_csv('MemberData/LukaRawDataJumping.csv')
BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')
df8 = pd.read_csv('MemberData/CJRawDataJumping.csv')

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


listOfJumpingData = pd.concat([df7,df8,BennettJumpingData])
listOfJumpingData['labels'] = 1
lisfOfCombinedData = pd.concat([listOfWalkingData,listOfJumpingData])
lisfOfCombinedData = lisfOfCombinedData.drop('Time (s)', axis=1)

window_size = 500

data_windows = []

for i in range(0, len(lisfOfCombinedData)-window_size+1, window_size):
    window = lisfOfCombinedData.iloc[i:i+window_size,:]
    windowNP = window.to_numpy()
    np.random.shuffle(windowNP)
    windowDf = pd.DataFrame(windowNP)
    data_windows.append(windowDf)

data_windows = data_windows[:500]
np.random.shuffle(data_windows)
data_list = []
label_list = []
for sublist in data_windows:
    data_df = pd.DataFrame(sublist)
    data_list.append(data_df.iloc[:, :4])
    label_list.append(data_df.iloc[:, 4])
# print(label_list)
#print(data_list)


# Split the segmented data into 90% train and 10% test
# num_train = int(0.9 * len(all_segments))
# train_segments = all_segments[:num_train]
# test_segments = all_segments[num_train:]

# dataList = data.iloc[:,0:5]
# dataList = dataList.iloc[:250000, :]

# labelList = data.iloc[:,-1]
# labelList = labelList.iloc[:250000]

X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# def normalizeData(data, windowSize):
#     q1 = data.quantile(0.20)
#     q3 = data.quantile(0.80)
#     iqr = q3 - q1
#     data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))] = np.nan

#     data.interpolate(method='linear', inplace=True)

#     data = data.rolling(windowSize, center=True).mean()

#     data.fillna(method='ffill', inplace=True)

#     x = data.values
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     dataNew = pd.DataFrame(x_scaled, columns=data.columns)

#     return dataNew

# Part 4 - Preprocessing
def data_processing(windows, w_size):
    # Create array for filtered data
    filtered_data = np.zeros((windows.shape[0], windows.shape[1]-w_size+1, windows.shape[2]))

    # Loop through each window and apply a moving average filter to each feature
    for i in range(windows.shape[0]):
        for j in range(windows.shape[2]):
            # Creating dataframes
            feature_df = pd.DataFrame(windows[i, :, j])

            # Apply MA filter
            feature_sma = feature_df.rolling(w_size).mean().values.ravel()

            # Discard the filtered NaN values
            feature_sma = feature_sma[w_size - 1:]

            # Store filtered data in array
            filtered_data[i, :, j] = feature_sma

    # Extract features
    feature_data = train_feature_extraction(filtered_data)

    # Create dataframes for further processing
    feature_dfs = [pd.DataFrame(feature_data[:, :, i]) for i in range(feature_data.shape[2])]

    # Using z score to remove outliers in each dataframe
    for df in feature_dfs:
        for i in range(df.shape[1]):
            column_data = df.iloc[:, i]
            z_scores = (column_data - column_data.mean())/column_data.std()
            column_data = column_data.mask(abs(z_scores) > 3, other=np.nan)  # Threshold at a z score of 3
            df.iloc[:, i] = column_data.fillna(filtered_data.mean())  # Fill NaN values with mean

    # Creating filtered feature array with labels for each measurement
    filtered_feature_data = np.concatenate(feature_dfs, axis=0)
    labels = np.concatenate([i * np.ones((feature_data.shape[0], 1)) for i in range(feature_data.shape[2])], axis=0)
    filtered_feature_data = np.hstack((filtered_feature_data, labels))

    return filtered_feature_data

# normalizedData_train = normalizeData(X_train, 5)
# normalizedData_test = normalizeData(X_test, 5)
def train_feature_extraction(windows):
    # Create an empty array to hold the feature vectors
    features = np.zeros((windows.shape[0], 11))

    # Iterate over each time window and extract the features
    for i in range(windows.shape[0]):
        # Extract the data from the window
        window_data = windows[i]

        # Compute the features
        max_val = np.max(window_data)
        min_val = np.min(window_data)
        range_val = max_val - min_val
        mean_val = np.mean(window_data)
        median_val = np.median(window_data)
        var_val = np.var(window_data)
        skew_val = skew(window_data)
        rms_val = np.sqrt(np.mean(window_data ** 2))
        kurt_val = np.mean((window_data - np.mean(window_data)) ** 4) / (np.var(window_data) ** 2)
        std_val = np.std(window_data)
        abs_energy = np.sum(np.square(window_data))

        # Store the features in the features array
        features[i, 0] = max_val
        features[i, 1] = min_val
        features[i, 2] = range_val
        features[i, 3] = mean_val
        features[i, 4] = median_val
        features[i, 5] = var_val
        features[i, 6] = skew_val
        features[i, 7] = rms_val
        features[i, 8] = kurt_val
        features[i, 9] = std_val
        features[i, 10] = abs_energy

    return features

def test_feature_extraction(windows):
    # Create an empty array to hold the feature vectors
    features = np.zeros((windows.shape[0], 11))

    # Iterate over each time window and extract the features
    for i in range(windows.shape[0]):
        # Extract the data from the window
        window_data = windows[i]

        # Compute the features
        max_val = np.max(window_data)
        min_val = np.min(window_data)
        range_val = max_val - min_val
        mean_val = np.mean(window_data)
        median_val = np.median(window_data)
        var_val = np.var(window_data)
        skew_val = skew(window_data)
        rms_val = np.sqrt(np.mean(window_data ** 2))
        kurt_val = np.mean((window_data - np.mean(window_data)) ** 4) / (np.var(window_data) ** 2)
        std_val = np.std(window_data)
        abs_energy = np.sum(np.square(window_data))

        # Store the features in the features array
        features[i, 0] = max_val
        features[i, 1] = min_val
        features[i, 2] = range_val
        features[i, 3] = mean_val
        features[i, 4] = median_val
        features[i, 5] = var_val
        features[i, 6] = skew_val
        features[i, 7] = rms_val
        features[i, 8] = kurt_val
        features[i, 9] = std_val
        features[i, 10] = abs_energy

    return features

# def extract_features(data, wsize):
#     features = []
#     xyz_data = data.iloc[:, 1:4].rolling(window=wsize)
#     features.append(xyz_data.mean())
#     features.append(xyz_data.std())
#     features.append(xyz_data.max())
#     features.append(xyz_data.min())
#     features.append(xyz_data.median())
#     features.append(xyz_data.var())
#     features.append(xyz_data.kurt())
#     features.append(xyz_data.skew())
#     features.append(xyz_data.max()-xyz_data.min())

#     # Append a column of zeros to the combined X, Y, and Z data
#     features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
#                                xyz_data.min(), xyz_data.median(), xyz_data.var(),  
#                                xyz_data.kurt(), xyz_data.skew(),(xyz_data.max()-xyz_data.min())
#                                ))

#     datFrame = pd.DataFrame(features)
#     return datFrame


window_size = 5
X_train = np.array(X_train)
X_test = np.array(X_test)
# Initialize the features array with the correct shape
train_features = np.zeros((X_train.shape[0], 11))

# Call the train_feature_extraction function
train_features = train_feature_extraction(train_features)

filteredTraining = data_processing(X_train, window_size)
filteredTest = data_processing(X_test, window_size)
print(filteredTraining)
# normalizedData_train = []
# # for window in X_train:
# #     df = pd.DataFrame(window)
# #     normalized_window = normalizeData(df, 5)
# #     normalizedData_train.append(normalized_window)
# X_train = np.array(X_train)
# normalizedData_train = data_processing(X_train, 5)
# # normalize each data window in X_test
# normalizedData_test = []
# # for window in X_test:
# #     df = pd.DataFrame(window)
# #     normalized_window = normalizeData(df, 5)
# #     normalizedData_test.append(normalized_window)
# normalizedData_train = data_processing(X_test, 5)

# features_train = []
# for window in normalizedData_train:
#     features_window = extract_features(window, 5)
#     features_window = features_window.dropna()
#     features_train.append(features_window)

# # extract features from each normalized data window in X_test
# features_test = []
# for window in normalizedData_test:
#     features_window = extract_features(window, 5)
#     features_window = features_window.dropna()
#     features_test.append(features_window)
# training_dataset = pd.DataFrame(np.hstack((filteredTraining, y_train)))
# filteredTraining = pd.DataFrame(filteredTraining)
# filteredTraining = filteredTraining.iloc[:,0:10]
# print(filteredTraining)
# X_train = np.concatenate(filteredTraining).reshape(-1, 27)
# # X_train = pd.DataFrame(X_train)
# X_test = np.concatenate(filteredTest).reshape(-1, 27)
# X_test = pd.DataFrame(X_test)

# print(X_train)
# print(X_test)
# print(features_train)
# X_train = np.array(features_train).reshape(-1, 27)
# X_test = np.array(features_test).reshape(-1, 27)
# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)
# print(X_train)
# print(X_test)
# y_train = np.array(y_train).ravel()
# y_test = np.array(y_test).ravel()
# y_train = y_train[:len(X_train)]
# y_test = y_test[:len(X_test)]

# print(features_train.isna().sum())

# # # Extract features from normalized training data
# train_features = pd.DataFrame()
# for j in range(0, len(normalizedData_train) - 500, 500):
#     df = extract_features(normalizedData_train.iloc[j:j+500], 500)
#     train_features = pd.concat([train_features, df])

# # Extract features from normalized testing data
# test_features = pd.DataFrame()
# for j in range(0, len(normalizedData_test) - 500, 500):
#     df = extract_features(normalizedData_test.iloc[j:j+500], 500)
#     test_features = pd.concat([test_features, df])



# print(train_features.isna().sum())
# train_features = train_features.iloc[499:,:]
# print(train_features.isna().sum())
# test_features = test_features.iloc[499:,:]
# print(train_features)
# print(test_features)

# train_features.fillna(method = 'ffill',inplace=True)
# test_features.fillna(method = 'ffill',inplace=True)

# print(train_features.isna().sum())

# y_train = y_train.iloc[:len(train_features)]
# y_test = y_test.iloc[:len(test_features)]
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)

# # with open('model.pkl', 'wb') as file:
# #      pickle.dump(clf, file)

# y_pred = dtc.predict(X_test)
# y_prob = dtc.predict_proba(X_test)
# # print(classification_report(y_test, y_pred))
# print('y_pred is:',y_pred)
# print('y_clf_prob is:',y_prob)

# acc = accuracy_score(y_test,y_pred)
# print('accuracy is:',acc)

# recall = recall_score(y_test,y_pred)
# print('recall is:',recall)

# cm = confusion_matrix(y_test,y_pred)
# cm_display = ConfusionMatrixDisplay(cm).plot()
# plt.show()

# f1Score = f1_score(y_test, y_pred)
# print('F1 Score is:', f1Score)

# # Plot the ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

# auc = roc_auc_score(y_test,y_prob[:,-1])
# print('the AUC is:', auc)

# featureData = pd.DataFrame()
# tempData = pd.DataFrame()
# for i in range(0,6):
#     for j in range(0, len(normalizedData[i]) - 500, 500):
#         df = extract_features_walking(normalizedData[i].iloc[j:j+500-1, :],10)
#         tempData = pd.concat([tempData,df])

# featureData = pd.concat([featureData,tempData]) 
# tempData = pd.DataFrame()
# for i in range(6,8):
   
#     for j in range(0, len(normalizedData[i]) - 500, 500):
#         df = extract_features_jumping(normalizedData[i].iloc[j:j+500-1, :],10)
#         tempData = pd.concat([tempData,df])
# featureData = pd.concat([featureData,tempData])
# featureData.interpolate(method = 'linear',inplace=True)


# X = featureData.iloc[:,0:27]
# y = featureData.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle = True, random_state=0)

# with h5py.File('./ProjectFileTest.hdf5', 'w') as f:
#     # Create sub groups for each member
#     for member_name, member_data in all_data.items():
#         member_group = f.create_group(member_name)
#         member_group.create_dataset('walking', data=member_data['walking'])
#         member_group.create_dataset('jumping', data=member_data['jumping'])

#     # Create a sub group for the dataset
#     dataset_group = f.create_group('dataset')


#     train_segments = pd.concat([X_train,y_train],axis=1)
#     test_segments = pd.concat([X_test,y_test],axis=1)

#     dataset_group.create_dataset('train',data=train_segments.values, dtype=train_segments.values.dtype)
#     dataset_group.create_dataset('test',data=test_segments.values, dtype=test_segments.values.dtype)

