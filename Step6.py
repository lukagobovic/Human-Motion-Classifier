from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import accuracy_score, recall_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, f1_score, confusion_matrix
from scipy.stats import skew
from sklearn.tree import DecisionTreeClassifier
import scikitplot as skplt
import pickle

# # Load data from HDF5 file
with h5py.File('ProjectFileTest.hdf5', 'r') as f:
    X_train = f['dataset/train'][:,0:27]
    y_train = f['dataset/train'][:,-1]
    X_test = f['dataset/test'][:,0:27]
    y_test = f['dataset/test'][:,-1]

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(dtc, file)

y_pred = dtc.predict(X_test)
y_prob = dtc.predict_proba(X_test)
print(y_prob.shape)
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
