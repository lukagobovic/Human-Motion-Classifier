import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import skew


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
window_size = 100

# Function to compute summary statistics for each axis over a window
def compute_stats(data):
  return [np.mean(data), np.std(data), np.min(data), np.max(data), np.ptp(data), np.median(data), np.var(data), skew(data)]


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
walking_data = pd.read_csv('Raw Data walking front pocket.csv')
running_data = pd.read_csv('Raw Data jumping .csv')
walking_data = walking_data.dropna() #<--- SECOND ISSUE
running_data = running_data.dropna() #<--- SECOND ISSUE

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
X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_labels, test_size=0.1,shuffle = True, random_state=0)

# Train a decision tree classifier on the training data
dtc = RandomForestClassifier()
dtc.fit(X_train, y_train)

# Test the model on the testing data and evaluate its performance
y_pred = dtc.predict(X_test)
# accuracy = np.mean(y_pred == y_test)
accuracy = accuracy_score(y_test, y_pred)

# new_data = pd.read_csv('Raw Data jumping .csv')
# prediction = lr.predict(new_data)
# print(f'Prediction: {prediction}')
print("Accuracy: {:.2f}%".format(accuracy * 100))
