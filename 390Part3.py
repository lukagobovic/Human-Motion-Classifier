import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# jumping_data = jumping_data[1000:2000,0:5]
walking_data = read_data('walking')
# walking_data = walking_data[1000:2000,0:5]

# Time series plot of acceleration magnitude for walking and jumping
fig, ax = plt.subplots(figsize=(12, 8))
plot_data(ax, walking_data, 'Acceleration Magnitude vs Time', 'blue', 'Walking')
plot_data(ax, jumping_data, 'Acceleration Magnitude vs Time', 'red', 'Jumping')
fig.tight_layout()
plt.show()

# Histogram of acceleration magnitudes for walking and jumping

# Scatter plot of x vs y, x vs z, and y vs z for both walking and jumping data
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes[0].scatter(walking_data[:, 1], walking_data[:, 2], color='blue', label='Walking', alpha=0.5)
axes[0].scatter(jumping_data[:, 1], jumping_data[:, 2], color='red', label='Jumping', alpha=0.5)
axes[0].set_title('Acceleration Y vs Acceleration X')
axes[0].set_xlabel('Acceleration X')
axes[0].set_ylabel('Acceleration Y')
axes[0].legend()

axes[1].scatter(walking_data[:, 1], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
axes[1].scatter(jumping_data[:, 1], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)
axes[1].set_title('Acceleration Z vs Acceleration X')
axes[1].set_xlabel('Acceleration X')
axes[1].set_ylabel('Acceleration Z')
axes[1].legend()

axes[2].scatter(walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
axes[2].scatter(jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)
axes[2].set_title('Acceleration Z vs Acceleration Y')
axes[2].set_xlabel('Acceleration Y')
axes[2].set_ylabel('Acceleration Z')
axes[2].legend()

# fig.tight_layout()
# plt.show()

# # 3D Scatter plot of x, y, and z accelerations for both walking and jumping dat
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(walking_data[:, 1], walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
# ax.scatter(jumping_data[:, 1], jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)

# ax.set_title('3D Scatter Plot of Acceleration')
# ax.set_xlabel('Acceleration X')
# ax.set_ylabel('Acceleration Y')
# ax.set_zlabel('Acceleration Z')
# ax.legend()

# plt.show()

#plotting windows
def plot_window(walking_data, jumping_data, window_start, window_size):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot walking data window
    walking_window = walking_data[window_start:window_start + window_size, :]
    plot_data(ax[0], walking_window, 'Walking: Acceleration Magnitude vs Time', 'blue', 'Walking')

    # Plot jumping data window
    jumping_window = jumping_data[window_start:window_start + window_size, :]
    plot_data(ax[1], jumping_window, 'Jumping: Acceleration Magnitude vs Time', 'red', 'Jumping')

    fig.tight_layout()
    plt.show()

window_start = 1000
window_size = 1500
plot_window(walking_data, jumping_data, window_start, window_size)

# # Plot acceleration vs. time for a sample of walking and jumping data
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax[0].plot(walking_data[:, 0], walking_data[:, 1], label='x')
ax[0].plot(walking_data[:, 0], walking_data[:, 2], label='y')
ax[0].plot(walking_data[:, 0], walking_data[:, 3], label='z')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Acceleration (m/s^2)')
ax[0].set_title('Walking Acceleration vs. Time')
ax[0].legend()

ax[1].plot(jumping_data[:, 0], jumping_data[:, 1], label='x')
ax[1].plot(jumping_data[:, 0], jumping_data[:, 2], label='y')
ax[1].plot(jumping_data[:, 0], jumping_data[:, 3], label='z')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Acceleration (m/s^2)')
ax[1].set_title('Jumping Acceleration vs. Time')
ax[1].legend()

# plt.show()

# Plot histograms of acceleration magnitudes for walking and jumping data
walking_mags = np.linalg.norm(walking_data[:, 1:4], axis=1)
jumping_mags = np.linalg.norm(jumping_data[:, 1:4], axis=1)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim([0, 30]) 
ax.hist(walking_mags.flatten(), bins=50, alpha=0.5, color='orange', label='walking')
ax.hist(jumping_mags.flatten(), bins=50, alpha=0.5, color='green', label='jumping')
ax.set_title('Histogram of Acceleration Magnitudes')
ax.set_xlabel('Acceleration Magnitude')
ax.set_ylabel('Frequency')
ax.legend()
fig.tight_layout()
plt.show()


