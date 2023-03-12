import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Load the sensor data (assuming it is in a CSV file with columns "accel_x", "accel_y", "accel_z")
data = pd.read_csv('sensor_data.csv')
acceleration_data = data[['accel_x', 'accel_y', 'accel_z']].values

# Define the sampling frequency and time step
fs = 800  # Sampling frequency (Hz)
dt = 1/fs  # Time step (s)

# Create a Kalman filter with 3 states (position, velocity, acceleration) and 3 measurement inputs (acceleration in x, y, z)
kf = KalmanFilter(dim_x=3, dim_z=3)

# Define the state transition matrix (assuming constant acceleration model)
kf.F = np.array([[1, dt, 0.5 * dt ** 2],
                 [0, 1, dt],
                 [0, 0, 1]])

# Define the measurement matrix (only acceleration measurements are available)
kf.H = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

# Estimate the measurement noise covariance matrix from the data
R = np.cov(acceleration_data.T)

# Estimate the process noise covariance matrix from the data
delta_acceleration_data = np.diff(acceleration_data, axis=0)
Q = np.cov(delta_acceleration_data.T)

# Set the measurement noise and process noise covariance matrices
kf.R = R
kf.Q = Q

# Initialize the state vector (assuming zero initial position, velocity and acceleration)
x0 = np.array([0, 0, 0])
kf.x = x0

# Initialize the covariance matrix (assuming high uncertainty in the initial state)
P0 = np.diag([1000 ** 2, 1000 ** 2, 1000 ** 2])
kf.P = P0

# Run the Kalman filter on the acceleration data
N = acceleration_data.shape[0]
filtered_acceleration = np.zeros((N, 3))
for i in range(N):
    kf.predict()
    kf.update(acceleration_data[i, :])
    filtered_acceleration[i, :] = kf.x[2]  # Only keep the filtered acceleration

# Plot the filtered acceleration data with the true acceleration data
times = np.arange(N) * dt

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(times, acceleration_data[:, 0], label='True')
axs[0].plot(times, filtered_acceleration[:, 0], label='Filtered')
axs[0].set_ylabel('Acceleration X (m/s^2)')
axs[0].legend()

axs[1].plot(times, acceleration_data[:, 1], label='True')
axs[1].plot(times, filtered_acceleration[:, 1], label='Filtered')
axs[1].set_ylabel('Acceleration Y (m/s^2)')
axs[1].legend()

axs[2].plot(times, acceleration_data[:, 2], label='True')
axs[2].plot(times, filtered_acceleration[:, 2], label='Filtered')
axs[2].set_ylabel('Acceleration Z (m/s^2)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()

plt.suptitle('Kalman filter for acceleration data from ADXL375 sensor')
plt.show()
