# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from scipy.stats import skew
from sklearn.tree import DecisionTreeClassifier
import pickle


# Load the data

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
listOfJumpingData['labels'] = 1


lisfOfCombinedData = pd.concat([listOfWalkingData,listOfJumpingData])

# Preprocessing
dataList = lisfOfCombinedData.iloc[:,1:4]
labelList = lisfOfCombinedData.iloc[:,-1]
dataList = dataList.iloc[:190000, :]
labelList = labelList.iloc[:190000]

# Z-score normalization to remove outliers
z_scores = (dataList - dataList.mean()) / dataList.std()
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
dataList = dataList[filtered_entries]
labelList = labelList[filtered_entries]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataList, labelList, test_size=0.1,shuffle = True, random_state=0)

# Normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)

# Feature extraction
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
for j in range(0, len(X_train) - 500, 500):
    df = extract_features(X_train.iloc[j:j+500], 500)
    train_features = pd.concat([train_features, df])

# Extract features from normalized testing data
test_features = pd.DataFrame()
for j in range(0, len(X_test) - 500, 500):
    df = extract_features(X_test.iloc[j:j+500], 500)
    test_features = pd.concat([test_features, df])

train_features = train_features.fillna(train_features.mean())
test_features = test_features.fillna(test_features.mean())

y_train = y_train.iloc[:len(train_features)]
y_test = y_test.iloc[:len(test_features)]

# Random Forest classifier
dtc = DecisionTreeClassifier()
dtc.fit(train_features, y_train)

with open('model.pkl', 'wb') as file:
     pickle.dump(dtc, file)

# Evaluate the model
y_pred = dtc.predict(test_features)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)
