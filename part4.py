import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def remove_outliers(data):
    all_dfs = []
    for i in range(data.shape[0]):
        x_df = pd.DataFrame(data[i, :, 0])
        y_df = pd.DataFrame(data[i, :, 1])
        z_df = pd.DataFrame(data[i, :, 2])
        total_df = pd.DataFrame(data[i, :, 3])
        
        # Remove outliers from each dataframe
        for df in [x_df, y_df, z_df, total_df]:
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))] = np.nan
        
        # Combine the dataframes back into one array
        combined_df = pd.concat([x_df, y_df, z_df, total_df], axis=1)
        all_dfs.append(combined_df.to_numpy())
    
    return np.array(all_dfs)

# Function to filter the data window by window
def filterData(data, wsize):
    filtered_data = np.zeros((data.shape[0], data.shape[1]-wsize+1, data.shape[2]))

    for i in range(data.shape[0]):
        x_df = pd.DataFrame(data[i, :, 0])
        y_df = pd.DataFrame(data[i, :, 1])
        z_df = pd.DataFrame(data[i, :, 2])
        total_df = pd.DataFrame(data[i, :, 3])

        # Remove outliers using interquartile range (IQR) method for each axis
        for df in [x_df, y_df, z_df, total_df]:
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))] = np.nan

        # Replace NaN values with the mean of the remaining values
        x_df.fillna(x_df.mean(), inplace=True)
        y_df.fillna(y_df.mean(), inplace=True)
        z_df.fillna(z_df.mean(), inplace=True)
        total_df.fillna(total_df.mean(), inplace=True)

        # print(np.sum(x_df.isna()).sum())

        x_sma = x_df.rolling(wsize).mean().values.ravel()
        y_sma = y_df.rolling(wsize).mean().values.ravel()
        z_sma = z_df.rolling(wsize).mean().values.ravel()
        total_sma = total_df.rolling(wsize).mean().values.ravel()

        # Discard the filtered NaN values
        x_sma = x_sma[wsize - 1:]
        y_sma = y_sma[wsize - 1:]
        z_sma = z_sma[wsize - 1:]
        total_sma = total_sma[wsize - 1:]

        # print(np.sum(np.isnan(x_sma)).sum())

        # Normalize the filtered data
        sc = StandardScaler()
        x_scaled = sc.fit_transform(x_sma.reshape(-1, 1)).ravel()
        y_scaled = sc.fit_transform(y_sma.reshape(-1, 1)).ravel()
        z_scaled = sc.fit_transform(z_sma.reshape(-1, 1)).ravel()
        total_scaled = sc.fit_transform(total_sma.reshape(-1, 1)).ravel()

        # print(np.sum(np.isnan(x_scaled)).sum())

        # Replace NaN values with linear interpolation
        x_clean = pd.Series(x_scaled).interpolate().values
        y_clean = pd.Series(y_scaled).interpolate().values
        z_clean = pd.Series(z_scaled).interpolate().values
        total_clean = pd.Series(total_scaled).interpolate().values

        # print(np.sum(np.isnan(x_clean)).sum())
        # print("-----------------------------")

        filtered_data[i, :, 0] = x_clean
        filtered_data[i, :, 1] = y_clean
        filtered_data[i, :, 2] = z_clean
        filtered_data[i, :, 3] = total_clean

    return filtered_data


# Read the dataset
with h5py.File('./ProjectFile.hdf5', 'r') as hdf:
    walking = hdf['dataset/train/walking'][:, :, 1:]
    jumping = hdf['dataset/train/jumping'][:, :, 1:]

# Filter the data with a specified window size
window_size = 10
# walking_filtered = remove_outliers(walking)
# print(walking_filtered.shape)
# jumping_filtered = remove_outliers(jumping)
# print(walking_filtered.shape)
walking_filtered = filterData(walking, window_size)
jumping_filtered = filterData(jumping, window_size)
print(walking_filtered.shape)
print(jumping_filtered.shape)


# Observe the filter; specify window index and activity (0=walking, 1=jumping)
window_index = 499
activity = 0
if activity == 0:
    prefilter = walking
    postfilter = walking_filtered
    activity_type = 'walking'
else:
    prefilter = jumping
    postfilter = jumping_filtered
    activity_type = 'jumping'

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the unfiltered data for specified window in the left subplot
ax1.plot(range(500), prefilter[window_index, :, 0], label='x acceleration')
ax1.plot(range(500), prefilter[window_index, :, 1], label='y acceleration')
ax1.plot(range(500), prefilter[window_index, :, 2], label='z acceleration')

# Set the title and axis labels for the left subplot
ax1.set_title(f'Unfiltered {activity_type} data for window {window_index+1}')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Acceleration (m/s^2)')

# Add a legend for the left subplot
ax1.legend()

# Plot the filtered data for the window in the right subplot
ax2.plot(range(500-window_size+1), postfilter[window_index, :, 0], label='x acceleration')
ax2.plot(range(500-window_size+1), postfilter[window_index, :, 1], label='y acceleration')
ax2.plot(range(500-window_size+1), postfilter[window_index, :, 2], label='z acceleration')


# Set the title and axis labels for the right subplot
ax2.set_title(f'Filtered {activity_type} data for window {window_index+1} using a window size of {window_size}')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Acceleration (m/s^2)')

# Add a legend for the right subplot
ax2.legend()

# Show the plot
plt.show()