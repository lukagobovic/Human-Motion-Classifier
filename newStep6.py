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

with h5py.File('ProjectFileTest.hdf5', 'r') as f:
    # Read train data into a list of dataframes
    X_train = [pd.DataFrame(f['dataset/train'][i, :, 1:5]) for i in range(f['dataset/train'].shape[0])]
    y_train = [pd.DataFrame(f['dataset/train'][i, :, 5]) for i in range(f['dataset/train'].shape[0])]

    X_test = [pd.DataFrame(f['dataset/test'][i, :, 1:5]) for i in range(f['dataset/test'].shape[0])]
    y_test = [pd.DataFrame(f['dataset/test'][i, :, 5]) for i in range(f['dataset/test'].shape[0])]




# print(X_train)
# print(X_test)

# print(y_train)
# print(y_test)
train_segments = X_train
test_segments = X_test
train_labels = y_train
test_labels = y_test
for df in train_segments:
    df.rename(columns={0: "x", 1: "y", 2: "z", 3: "total_acceleration"}, inplace=True)

for df in test_segments:
    df.rename(columns={0: "x", 1: "y", 2: "z", 3: "total_acceleration"}, inplace=True)

for df in train_labels:
    df.rename(columns={0: "activity"}, inplace=True)
    
for df in test_labels:
    df.rename(columns={0: "activity"}, inplace=True)

# # Preprocess the data
def preprocess_data(data):
  # Remove outliers
  data = data[(np.abs(data.x - data.x.mean()) / data.x.std()) < 2.5]
  data = data[(np.abs(data.y - data.y.mean()) / data.y.std()) < 2.5]
  data = data[(np.abs(data.z - data.z.mean()) / data.z.std()) < 2.5]

  # Normalize the data
  data = (data - data.mean()) / data.std()

  # Extract features
  features = [
    np.max(data.x),
    np.min(data.x),
    np.ptp(data.x),
    np.mean(data.x),
    np.median(data.x),
    np.var(data.x),
    np.std(data.x),
    np.max(data.y),
    np.min(data.y),
    np.ptp(data.y),
    np.mean(data.y),
    np.median(data.y),
    np.var(data.y),
    np.std(data.y),
    np.max(data.z),
    np.min(data.z),
    np.ptp(data.z),
    np.mean(data.z),
    np.median(data.z),
    np.var(data.z),
    np.std(data.z),
    np.max(data.total_acceleration),
    np.min(data.total_acceleration),
    np.ptp(data.total_acceleration),
    np.mean(data.total_acceleration),
    np.median(data.total_acceleration),
    np.var(data.total_acceleration),
    np.std(data.total_acceleration),
  ]

  return features

train_features = [preprocess_data(segment) for segment in train_segments]
train_labels = [segment.activity.values[0] for segment in train_labels]

test_features = [preprocess_data(segment) for segment in test_segments]
test_labels = [segment.activity.values[0] for segment in test_labels]

# Train a logistic regression model
model = DecisionTreeClassifier()
model.fit(train_features, train_labels)

with open('model.pkl', 'wb') as file:
      pickle.dump(model, file)

# Test the model
pred_labels = model.predict(test_features)
y_prob = model.predict_proba(test_features)
accuracy = accuracy_score(test_labels, pred_labels)
print('Accuracy:', accuracy)
print(classification_report(test_labels, pred_labels))
recall = recall_score(test_labels,pred_labels)
print('recall is:',recall)

cm = confusion_matrix(test_labels,pred_labels)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

f1Score = f1_score(test_labels, pred_labels)
print('F1 Score is:', f1Score)

# # Plot the ROC curve
# fpr, tpr, thresholds = roc_curve(test_labels, y_prob[:,1])
# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

auc = roc_auc_score(test_labels,y_prob[:,-1])
print('the AUC is:', auc)