import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score, f1_score
from matplotlib import pyplot as plt
import h5py
from scipy.stats import skew, kurtosis
from sklearn.model_selection import learning_curve
import seaborn as sns

with h5py.File('ProjectFile.hdf5', 'r') as f:
    # Read train data into a list of dataframes
    X_train = [pd.DataFrame(f['dataset/train'][i, :, 0:4]) for i in range(f['dataset/train'].shape[0])]
    y_train = [pd.DataFrame(f['dataset/train'][i, :, 4]) for i in range(f['dataset/train'].shape[0])]

    X_test = [pd.DataFrame(f['dataset/test'][i, :, 0:4]) for i in range(f['dataset/test'].shape[0])]
    y_test = [pd.DataFrame(f['dataset/test'][i, :, 4]) for i in range(f['dataset/test'].shape[0])]

train_segments = X_train
test_segments = X_test
train_labels = y_train
test_labels = y_test
# Renaming labels to remain consistent
for df in train_segments:
    df.rename(columns={0: "x", 1: "y", 2: "z", 3: "total_acceleration"}, inplace=True)

for df in test_segments:
    df.rename(columns={0: "x", 1: "y", 2: "z", 3: "total_acceleration"}, inplace=True)

for df in train_labels:
    df.rename(columns={0: "activity"}, inplace=True)
    
for df in test_labels:
    df.rename(columns={0: "activity"}, inplace=True)

# Extract the features via a rolling window. 
# This results in a list of 100 features for every 5 second segment (10 means, 10 mins...)
def preprocess_data(data):
  window_size = 50
  num_windows = len(data) // window_size
  features = []
  for i in range(num_windows):
    window_data = data.iloc[i*window_size:(i+1)*window_size]
    window_features = [
        np.max(window_data.x),
        np.min(window_data.x),
        np.ptp(window_data.x),
        np.mean(window_data.x),
        np.median(window_data.x),
        skew(window_data.x),
        np.var(window_data.x),
        np.std(window_data.x),
        kurtosis(window_data.x),
        np.sqrt(np.mean(window_data.x ** 2)),
        np.max(window_data.y),
        np.min(window_data.y),
        np.ptp(window_data.y),
        np.mean(window_data.y),
        np.median(window_data.y),
        skew(window_data.y),
        np.var(window_data.y),
        np.std(window_data.y),
        kurtosis(window_data.y),
        np.sqrt(np.mean(window_data.y ** 2)),
        np.max(window_data.z),
        np.min(window_data.z),
        np.ptp(window_data.z),
        np.mean(window_data.z),
        np.median(window_data.z),
        skew(window_data.z),
        np.var(window_data.z),
        np.std(window_data.z),
        kurtosis(window_data.z),
        np.sqrt(np.mean(window_data.z ** 2)),
        np.max(window_data.total_acceleration),
        np.min(window_data.total_acceleration),
        np.ptp(window_data.total_acceleration),
        np.mean(window_data.total_acceleration),
        np.median(window_data.total_acceleration),
        skew(window_data.total_acceleration),
        np.var(window_data.total_acceleration),
        np.std(window_data.total_acceleration),
        kurtosis(window_data.total_acceleration),
        np.sqrt(np.mean(window_data.total_acceleration ** 2))
    ]
    features.append(window_features)

  return features

#Ensures that the label shapes match the feature shapes
train_labels = np.concatenate([label[:((int)(500/window_size))] for label in train_labels])
test_labels = np.concatenate([label[:((int)(500/window_size))] for label in test_labels])
train_labels = train_labels.ravel()
test_labels = test_labels.ravel()

#Extract features
train_features = np.concatenate([preprocess_data(segment) for segment in train_segments])
test_features = np.concatenate([preprocess_data(segment) for segment in test_segments])

# Train the model
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
model = RandomForestClassifier(max_depth=5, random_state=42)
# l_reg = LogisticRegression(max_iter=10000,random_state=42)
# model = make_pipeline(StandardScaler(),l_reg)
model.fit(train_features, train_labels)

# Save model to pickle to be used in GUI
with open('model.pkl', 'wb') as file:
      pickle.dump(model, file)

# scores = cross_val_score(model, train_features+test_features, train_labels+test_labels, cv=5)
# print(scores)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# Calculate the learning curves using the learning_curve function
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(model, train_features, train_labels, cv=5, train_sizes=train_sizes)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

sns.set_style('whitegrid')

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training set size')
plt.ylabel('Accuracy score')
plt.title('Random Forest ClassifierLearning Curves')
plt.legend()
plt.show()

# Test the model
pred_labels = model.predict(test_features)
y_prob = model.predict_proba(test_features)

print(classification_report(test_labels, pred_labels))

train_accuracy = model.score(train_features, train_labels)
print("Training Accuracy:", train_accuracy)

# compute the accuracy on the test set
test_accuracy = model.score(test_features, test_labels)
print("Test Accuracy:", test_accuracy)

recall = recall_score(test_labels,pred_labels)
print('recall is:',recall)

cm = confusion_matrix(test_labels,pred_labels)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

f1Score = f1_score(test_labels, pred_labels)
print('F1 Score is:', f1Score)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(test_labels, y_prob[:,1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc = roc_auc_score(test_labels,y_prob[:,-1])
print('the AUC is:', auc)
