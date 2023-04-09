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


window_size = 500 # assuming the sampling rate is 50 Hz
data_windows = []
for i in range(0, len(lisfOfCombinedData)-window_size+1, window_size):
    window = lisfOfCombinedData.iloc[i:i+window_size,:]
    windowNP = window.to_numpy()
    np.random.shuffle(windowNP)
    windowDf = pd.DataFrame(windowNP)
    data_windows.append(windowDf)


data = pd.concat(data_windows, ignore_index=True)
print(data)

dataList = data.iloc[:,1:5]
dataList = dataList.iloc[:250000, :]
print(dataList)
labelList = data.iloc[:,-1]
labelList = labelList.iloc[:250000]
print(labelList)
X_train, X_test, y_train, y_test = train_test_split(dataList, labelList, test_size=0.1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

def normalizeData(data, windowSize):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))] = np.nan

    data.interpolate(method='linear', inplace=True)

    data = data.rolling(windowSize, center=True).mean()

    data.fillna(method='ffill', inplace=True)

    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dataNew = pd.DataFrame(x_scaled, columns=data.columns)

    return dataNew


normalizedData_train = normalizeData(X_train, 10)
normalizedData_test = normalizeData(X_test, 10)


def extract_features(data, wsize):
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
    features.append(xyz_data.max()-xyz_data.min())

    # Append a column of zeros to the combined X, Y, and Z data
    features = np.hstack((xyz_data.mean(), xyz_data.std(), xyz_data.max(),
                               xyz_data.min(), xyz_data.median(), xyz_data.var(),  
                               xyz_data.kurt(), xyz_data.skew(),(xyz_data.max()-xyz_data.min())
                               ))

    datFrame = pd.DataFrame(features)
    return datFrame

# Extract features from normalized training data
train_features = pd.DataFrame()
for j in range(0, len(normalizedData_train) - 500, 500):
    df = extract_features(normalizedData_train.iloc[j:j+500], 500)
    train_features = pd.concat([train_features, df])

# Extract features from normalized testing data
test_features = pd.DataFrame()
for j in range(0, len(normalizedData_test) - 500, 500):
    df = extract_features(normalizedData_test.iloc[j:j+500], 500)
    test_features = pd.concat([test_features, df])

train_features.fillna(method = 'ffill',inplace=True)
test_features.fillna(method = 'ffill',inplace=True)

print(train_features.isna().sum())

y_train = y_train.iloc[:len(train_features)]
y_test = y_test.iloc[:len(test_features)]

l_reg = LogisticRegression(max_iter=10000)
scaler = StandardScaler()
clf = make_pipeline(scaler,l_reg)
clf.fit(train_features, y_train)

with open('model.pkl', 'wb') as file:
     pickle.dump(clf, file)

y_pred = clf.predict(test_features)
y_prob = clf.predict_proba(test_features)
print(classification_report(y_test, y_pred))
print('y_pred is:',y_pred)
print('y_clf_prob is:',y_prob)

acc = accuracy_score(y_test,y_pred)
print('accuracy is:',acc)

recall = recall_score(y_test,y_pred)
print('recall is:',recall)

cm = confusion_matrix(y_test,y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

f1Score = f1_score(y_test, y_pred)
print('F1 Score is:', f1Score)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc = roc_auc_score(y_test,y_prob[:,-1])
print('the AUC is:', auc)

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

