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
import pickle



# walking_data = pd.read_csv('5_sec_walking_front.csv')
# walking_data = walking_data.iloc[:,1:4]
# walking_data['activity'] = 0

# running_data = pd.read_csv('5_sec_jumping_front.csv')
# running_data = running_data.iloc[:,1:4]
# running_data['activity'] = 1

# data = pd.concat([walking_data, running_data],ignore_index=True)

# dataset_walking = pd.read_csv("5_sec_walking_front.csv")
# dataset_walking = dataset_walking.iloc[:,1:5]
# datawalking = dataset_walking.iloc[:, 0:-1]
# datawalking['activity'] = 0

# dataset_jumping = pd.read_csv("5_sec_jumping_front.csv")
# dataset_jumping = dataset_jumping.iloc[:,1:5]
# datajumping = dataset_jumping.iloc[:, 0:-1]
# datajumping['activity'] = 1

# df = pd.concat([datawalking, datajumping], ignore_index=True)

# X = df.iloc[:,0:3]
# y = df['activity']
# #data['activity'] = data.apply(lambda row: 'walking' if 'walking' in row['filename'] else 'running', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)

# score = lr.score(X_test, y_test)
# print(f'Model accuracy: {score:.2f}')

# new_data = pd.read_csv('Raw Data jumping .csv')
# prediction = lr.predict(new_data)
# print(f'Prediction: {prediction}')

# Define the time window for computing summary statistics
window_size = 8

# Function to compute summary statistics for each axis over a window
def compute_stats(data):
  return [np.mean(data), np.std(data), np.min(data), np.max(data), np.median(data), np.var(data), skew(data),np.sqrt(np.mean(data**2))]


# Function to extract features from the accelerometer data
def extract_features(data):
    # Create empty list to store features for each window
    features = []
    # Iterate over each window of data
    for i in range(0, len(data) - window_size, window_size):
        # Extract the x, y, z accelerometer data for the current window
        window_data = data[i:i+window_size, :]
        x_data = window_data[:, 0]
        y_data = window_data[:, 1]
        z_data = window_data[:, 2]
        # Compute summary statistics for each axis over the window
        x_stats = compute_stats(x_data)
        y_stats = compute_stats(y_data)
        z_stats = compute_stats(z_data)
        # Append the summary statistics to the features list
        features.append(x_stats + y_stats + z_stats)
    # Convert the features list to a numpy array and return it
    return np.array(features)

# Load the two datasets into Pandas dataframes using the read_csv method.
walking_data = pd.read_csv('5_sec_walking_front.csv')
running_data = pd.read_csv('5_sec_jumping_front.csv')
walking_data = walking_data.dropna() 
running_data = running_data.dropna()

# Preprocess the data by extracting features from the accelerometer data
walking_features = extract_features(walking_data.values[:, 1:])
running_features = extract_features(running_data.values[:, 1:])

# Create a new dataframe that combines the two datasets and adds a new binary column called "label"
walking_labels = np.zeros(len(walking_features), dtype=int)
running_labels = np.ones(len(running_features), dtype=int)
combined_features = np.vstack((walking_features, running_features))
combined_labels = np.concatenate((walking_labels, running_labels))
combined_data = np.column_stack((combined_features, combined_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, Y_test = train_test_split(combined_features, combined_labels, test_size=0.1,shuffle = True, random_state=0)

l_reg = LogisticRegression(max_iter=10000)
scaler = StandardScaler()
clf = make_pipeline(scaler,l_reg)
clf.fit(X_train, y_train)

pickle.dump(clf,open('model.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
df = pd.read_csv('Raw Data jumping .csv')
scaled_data = extract_features(df.values[:, 1:])
predictions = model.predict(scaled_data)
print(f'Prediction: {predictions}')

num_ones = np.count_nonzero(predictions)
print(predictions.size)
print("Number of ones:", num_ones)
accuracy1 = accuracy_score(Y_test, predictions)
print("Accuracy: {:.2f}%".format(accuracy1 * 100))


# Test the model on the testing data and evaluate its performance
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
accuracy = accuracy_score(Y_test, y_pred)

print(X_test.shape)
# new_data = pd.read_csv('Raw Data jumping .csv')
# prediction = clf.predict(new_data)
# print(f'Prediction: {prediction}')
print("Accuracy: {:.2f}%".format(accuracy * 100))

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
