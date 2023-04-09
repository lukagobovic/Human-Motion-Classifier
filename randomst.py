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


df1 = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv',nrows = 30005)
df2 = pd.read_csv('MemberData/LukaRawDataWalkingJacket.csv',nrows = 30005)
df3 = pd.read_csv('MemberData/LukaRawDataBackPocketWalking.csv',nrows = 30005)

df4 = pd.read_csv('MemberData/CJRawDataFrontPocketWalking.csv',nrows = 30005)
df5 = pd.read_csv('MemberData/CJRawDataJacketWalking.csv',nrows = 30005)
df6 = pd.read_csv('MemberData/CJRawDataBackPocketWalking.csv',nrows = 30005)
listOfWalkingData = pd.DataFrame()
listOfWalkingData = pd.concat([df1,df2,df3,df4,df5,df6])
listOfWalkingData['activity'] = 0

listOfJumpingData = pd.DataFrame()
df7 = pd.read_csv('MemberData/LukaRawDataJumping.csv')
# BennettJumpingData = pd.read_csv('MemberData/BennettRawDataJumping.csv')
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


listOfJumpingData = pd.concat([df7,df8])
listOfJumpingData['activity'] = 1


lisfOfCombinedData = pd.concat([listOfWalkingData,listOfJumpingData])

# # Load data from a CSV file
# dataWalking = pd.read_csv('MemberData/LukaRawDataFrontPocketWalking.csv')
# dataJumping = pd.read_csv('MemberData/LukaRawDataJumping.csv')
# dataWalking['activity'] = 0
# dataJumping['activity'] = 1
# Concatenate the data into a single DataFrame
data = lisfOfCombinedData

# Define a window size in seconds
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

with h5py.File('./ProjectFileTest.hdf5', 'w') as f:
    # Create sub groups for each member
    for member_name, member_data in all_data.items():
        member_group = f.create_group(member_name)
        member_group.create_dataset('walking', data=member_data['walking'])
        member_group.create_dataset('jumping', data=member_data['jumping'])

    # Create a sub group for the dataset
    dataset_group = f.create_group('dataset')

    dataset_group.create_dataset('train',data=[s.values for s in train_segments])
    dataset_group.create_dataset('test',data=[s.values for s in test_segments])




# # Preprocess the data
# def preprocess_data(data):
#   # Remove outliers

#   data = data.drop(columns=['activity'])

#   data = data[(np.abs(data.x - data.x.mean()) / data.x.std()) < 3]
#   data = data[(np.abs(data.y - data.y.mean()) / data.y.std()) < 3]
#   data = data[(np.abs(data.z - data.z.mean()) / data.z.std()) < 3]

#   # Normalize the data
#   data = (data - data.mean()) / data.std()

#   # Extract features
#   features = [
#     np.max(data.x),
#     np.min(data.x),
#     np.ptp(data.x),
#     np.mean(data.x),
#     np.median(data.x),
#     np.var(data.x),
#     np.std(data.x),
#     np.max(data.y),
#     np.min(data.y),
#     np.ptp(data.y),
#     np.mean(data.y),
#     np.median(data.y),
#     np.var(data.y),
#     np.std(data.y),
#     np.max(data.z),
#     np.min(data.z),
#     np.ptp(data.z),
#     np.mean(data.z),
#     np.median(data.z),
#     np.var(data.z),
#     np.std(data.z),
#     np.max(data.total_acceleration),
#     np.min(data.total_acceleration),
#     np.ptp(data.total_acceleration),
#     np.mean(data.total_acceleration),
#     np.median(data.total_acceleration),
#     np.var(data.total_acceleration),
#     np.std(data.total_acceleration),
#   ]

#   return features

# train_features = [preprocess_data(segment) for segment in train_segments]
# train_labels = [segment.activity.values[0] for segment in train_segments]

# test_features = [preprocess_data(segment) for segment in test_segments]
# test_labels = [segment.activity.values[0] for segment in test_segments]

# # Train a logistic regression model
# model = DecisionTreeClassifier()
# model.fit(train_features, train_labels)

# with open('model.pkl', 'wb') as file:
#       pickle.dump(model, file)

# # Test the model
# pred_labels = model.predict(test_features)
# y_prob = model.predict_proba(test_features)
# accuracy = accuracy_score(test_labels, pred_labels)
# print('Accuracy:', accuracy)
# print(classification_report(test_labels, pred_labels))
# recall = recall_score(test_labels,pred_labels)
# print('recall is:',recall)

# cm = confusion_matrix(test_labels,pred_labels)
# cm_display = ConfusionMatrixDisplay(cm).plot()
# plt.show()

# f1Score = f1_score(test_labels, pred_labels)
# print('F1 Score is:', f1Score)

# # Plot the ROC curve
# fpr, tpr, thresholds = roc_curve(test_labels, y_prob[:,1])
# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

# auc = roc_auc_score(test_labels,y_prob[:,-1])
# print('the AUC is:', auc)
