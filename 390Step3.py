import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def read_data(activity):
    with h5py.File('./ProjectFile.hdf5', 'r') as f:
        member_walking_data = f['Luka/walking'][:]
        member_jumping_data = f['Luka/jumping'][:]
    if activity == 'walking':
        return member_walking_data
    elif activity == 'jumping':
        return member_jumping_data
    else:
        raise ValueError('Invalid activity value')

def plot_data(ax, data, title, color, label):
    time = data[:, 0]
    acceleration_x = data[:, 1]
    acceleration_y = data[:, 2]
    acceleration_z = data[:, 3]
    acceleration_total = data[:,4]
    ax.plot(time, acceleration_total, color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration Magnitude')
    ax.legend()

# Read the data
jumping_data = read_data('jumping')
walking_data = read_data('walking')

jumping_data_df = pd.DataFrame(jumping_data)
jumping_data_df = jumping_data_df.iloc[:,1:]
walking_data_df = pd.DataFrame(walking_data)
walking_data_df = walking_data_df.iloc[:,1:]
jumping_data_df.rename(columns={1: "x", 2: "y", 3: "z", 4: "total_acceleration"}, inplace=True)
walking_data_df.rename(columns={1: "x", 2: "y", 3: "z", 4: "total_acceleration"}, inplace=True)


#plotting windows
# fig, ax = plt.subplots(2, 1, figsize=(12, 8))

#     # Plot walking data window
# walking_window = walking_data[1000:2500, :]
# plot_data(ax[0], walking_window, 'Walking: Acceleration Magnitude vs Time', 'blue', 'Walking')

#     # Plot jumping data window
# jumping_window = jumping_data[1000:2500, :]
# plot_data(ax[1], jumping_window, 'Jumping: Acceleration Magnitude vs Time', 'red', 'Jumping')

# fig.tight_layout()
# plt.show()


# # # Plot acceleration vs. time for a sample of walking and jumping data
df = walking_data_df
Q1 = df.quantile(0.30)
Q3 = df.quantile(0.70)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]

  #Interpolate the missing values
df.interpolate(method='linear', inplace=True)
  
  #Rolling average filter
dfWalk = df.rolling(10).mean().dropna()

# scaler = MinMaxScaler()
# df = scaler.fit_transform(df)
# dfWalk = pd.DataFrame(df)

df = jumping_data_df
Q1 = df.quantile(0.30)
Q3 = df.quantile(0.70)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]

  #Interpolate the missing values
df.interpolate(method='linear', inplace=True)
  
  #Rolling average filter
dfJump = df.rolling(10).mean().dropna()
    
# scaler = MinMaxScaler()
# df = scaler.fit_transform(df)
# dfJump = pd.DataFrame(df)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax[0].plot(dfWalk.iloc[1000:3000, 1], label='x')
ax[0].plot(dfWalk.iloc[1000:3000, 2], label='y')
ax[0].plot(dfWalk.iloc[1000:3000, 3], label='z')
ax[0].set_ylabel('Acceleration (m/s^2)')
ax[0].set_title('Walking Acceleration vs. Time')
ax[0].legend()

ax[1].plot(dfJump.iloc[1000:3000, 1], label='x')
ax[1].plot(dfJump.iloc[1000:3000, 2], label='y')
ax[1].plot(dfJump.iloc[1000:3000, 3], label='z')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Acceleration (m/s^2)')
ax[1].set_title('Jumping Acceleration vs. Time')
ax[1].legend()

plt.show()


# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
# ax[0].plot(walking_data[1000:2000, 0], walking_data[1000:2000, 1], label='x')
# ax[0].plot(walking_data[1000:2000, 0], walking_data[1000:2000, 2], label='y')
# ax[0].plot(walking_data[1000:2000, 0], walking_data[1000:2000, 3], label='z')
# ax[0].set_xlabel('Time (s)')
# ax[0].set_ylabel('Acceleration (m/s^2)')
# ax[0].set_title('Walking Acceleration vs. Time')
# ax[0].legend()

# ax[1].plot(jumping_data[1000:2000, 0], jumping_data[1000:2000, 1], label='x')
# ax[1].plot(jumping_data[1000:2000, 0], jumping_data[1000:2000, 2], label='y')
# ax[1].plot(jumping_data[1000:2000, 0], jumping_data[1000:2000, 3], label='z')
# ax[1].set_xlabel('Time (s)')
# ax[1].set_ylabel('Acceleration (m/s^2)')
# ax[1].set_title('Jumping Acceleration vs. Time')
# ax[1].legend()

# # plt.show()


#HEATMAP 
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 8))
# corr_matrix_jumping = jumping_data_df.corr()
# corr_matrix_walking = walking_data_df.corr()

# # Create heatmap using seaborn library on the first axis
# sns.heatmap(corr_matrix_jumping, annot=True, cmap='coolwarm', ax=ax1)
# ax1.set_title('Jumping Correlation Heatmap')
# sns.heatmap(corr_matrix_walking, annot=True, cmap='coolwarm', ax=ax2)
# ax2.set_title('Walking Correlation Heatmap')
# fig.suptitle('Correlation Between Axes Heatmap', fontsize=16)
# fig.tight_layout()
# plt.show()


# fig =  sns.displot(jumping_data_df, kind="kde", bw_adjust=.25, fill = True)
# plt.title("Jumping Acceleration Magnitude Density")
# plt.xlabel("Magnitude")
# plt.ylabel("Density") 
# fig.tight_layout()
# plt.show()

# fig =  sns.displot(walking_data_df, kind="kde", bw_adjust=.25, fill = True)
# plt.title("Walking Acceleration Magnitude Density")
# plt.xlabel("Magnitude")
# plt.ylabel("Density") 
# fig.tight_layout()
# plt.show()


# sns.displot(jumping_data_df, x="x", y="y")
# plt.show()

# # Scatter plot of x vs y, x vs z, and y vs z for both walking and jumping data

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
# axes[0].scatter(walking_data[:, 1], walking_data[:, 2], color='blue', label='Walking', alpha=0.5)
# axes[0].scatter(jumping_data[:, 1], jumping_data[:, 2], color='red', label='Jumping', alpha=0.2)
# axes[0].set_title('Acceleration Y vs Acceleration X')
# axes[0].set_xlabel('Acceleration X')
# axes[0].set_ylabel('Acceleration Y')
# axes[0].legend()

# axes[1].scatter(walking_data[:, 1], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
# axes[1].scatter(jumping_data[:, 1], jumping_data[:, 3], color='red', label='Jumping', alpha=0.2)
# axes[1].set_title('Acceleration Z vs Acceleration X')
# axes[1].set_xlabel('Acceleration X')
# axes[1].set_ylabel('Acceleration Z')
# axes[1].legend()

# axes[2].scatter(walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
# axes[2].scatter(jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.2)
# axes[2].set_title('Acceleration Z vs Acceleration Y')
# axes[2].set_xlabel('Acceleration Y')
# axes[2].set_ylabel('Acceleration Z')
# axes[2].legend()

# fig.tight_layout()
# plt.show()

# Pair Plots
# g = sns.pairplot(walking_data_df)
# g.fig.suptitle("Walking Pair Plot", y=1.08) # y= some height>1
# fig.tight_layout()
# plt.show()

# fig = sns.pairplot(jumping_data_df)
# g.fig.suptitle("Jumping Pair Plot", y=1.08) # y= some height>1
# fig.tight_layout()
# plt.show()

