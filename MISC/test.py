import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style 
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import FunctionTransformer

style.use('ggplot')

dataset = pd.read_csv("5_sec_walking_front.csv")
data = dataset.iloc[0:,1:4]
# print(data.isna().sum())


mean = data.rolling(31).mean().dropna()
std = data.rolling(31).std().dropna()
max_val = data.rolling(31).max().dropna()
min_val = data.rolling(31).min().dropna()

features = {'mean': mean, 'std': std, 'max': max_val, 'min': min_val}
print(features)

#removing outliers by using percentiles. 
#This essentially will remove all points that are outside the lower 25% and upper 75%
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]

print(data)

window_size = 5
data5 = data.rolling(window_size, center=True).mean()

window_size = 31
data31 = data.rolling(window_size, center=True).mean()

window_size = 51
data51 = data.rolling(window_size, center=True).mean()

# fig, ax = plt.subplots()

plt.plot(std.iloc[0:,1], label = 'Unfiltered')
#ax.plot(data5, label='Window size 5')
#ax.plot(data31, label='Window size 31')
#ax.plot(data51, label='Window size 51')

# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
# ax.legend()

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, f1_score


dataset = pd.read_csv("winequalityN-lab6.csv")
newdataset = dataset.iloc[:,1:13]
newdataset.loc[newdataset['quality'] <= 5, 'quality'] = 0
newdataset.loc[newdataset['quality'] >= 6, 'quality'] = 1
data = newdataset.iloc[:, 0:-1]
labels = newdataset.iloc[:,-1]

print(data)
print(labels)

X_train , X_test, Y_train, Y_test = train_test_split(data,labels,test_size = 0.1,shuffle = True, random_state= 0)

scaler = StandardScaler()

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(),l_reg)

clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
print('y_pred is:',y_pred)
print('y_clf_prob is:',y_clf_prob)

acc = accuracy_score(Y_test,y_pred)
print('accuracy is:',acc)

recall = recall_score(Y_test,y_pred)
print('recall is:',recall)

cm = confusion_matrix(Y_test,y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

f1Score = f1_score(Y_test, y_pred)
print('F1 Score is:', f1Score)

fpr,tpr,_ = roc_curve(Y_test, y_clf_prob[:,1],pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr,tpr=tpr).plot()
plt.show()

auc = roc_auc_score(Y_test,y_clf_prob[:,-1])
print('the AUC is:', auc)