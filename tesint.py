
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import recall_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, f1_score

# Load the walking data into a dataframe
df_walking = pd.read_csv('5_sec_walking_front.csv', names=['time', 'x', 'y', 'z', 'total_acceleration'])
# Add a new column to indicate the activity (walking = 0)
df_walking['activity'] = 0
# Load the jumping data into a dataframe
df_jumping = pd.read_csv('5_sec_jumping_front.csv', names=['time', 'x', 'y', 'z', 'total_acceleration'])
# Add a new column to indicate the activity (jumping = 1)
df_jumping['activity'] = 1
# Concatenate the two dataframes into a single dataframe
df = pd.concat([df_walking, df_jumping], ignore_index=True)

df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)
df['z'] = df['z'].astype(float)
df['total_acceleration'] = df['total_acceleration'].astype(float)



X = df[['x', 'y', 'z', 'total_acceleration']]
y = df['activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Create an SVM classifier
clf = SVC(kernel='linear', C=1)

# Train the classifier on the training data
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_matrix)

#print(data)
##X_train , X_test, Y_train, Y_test = train_test_split(data,labels,test_size = 0.1,shuffle = True, random_state= 0)



# scaler = StandardScaler()

# l_reg = LogisticRegression(max_iter=10000)
# clf = make_pipeline(StandardScaler(),l_reg)

# clf.fit(X_train,Y_train)

# y_pred = clf.predict(X_test)
# y_clf_prob = clf.predict_proba(X_test)
# print('y_pred is:',y_pred)
# print('y_clf_prob is:',y_clf_prob)

# acc = accuracy_score(Y_test,y_pred)
# print('accuracy is:',acc)

# recall = recall_score(Y_test,y_pred)
# print('recall is:',recall)

# cm = confusion_matrix(Y_test,y_pred)
# cm_display = ConfusionMatrixDisplay(cm).plot()
# plt.show()

# f1Score = f1_score(Y_test, y_pred)
# print('F1 Score is:', f1Score)

# fpr,tpr,_ = roc_curve(Y_test, y_clf_prob[:,1],pos_label=clf.classes_[1])
# roc_display = RocCurveDisplay(fpr=fpr,tpr=tpr).plot()
# plt.show()

# auc = roc_auc_score(Y_test,y_clf_prob[:,-1])
# print('the AUC is:', auc)