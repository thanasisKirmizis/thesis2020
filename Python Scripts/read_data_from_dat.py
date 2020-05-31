
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
import struct
import datetime
from functions_suite import filter_gravity_effect, preproc_signal, calc_sess_metrics

# Definitions
TIMESTAMP_OF_CLAP_IN_VIDEO = -5840   # in millis
CUTOFF = 1                          # in Hz
FS = 50                             # in Hz
TIME_CONV = 10**3                  # 10**9 for nanos, 10**3 for millis
IS_IN_NANOS = False                  # True for nanos, False for millis

# Initializations
accel_data = []
accel_time_data = []
gyro_data = []
gyro_time_data = []
magnet_data = []
magnet_time_data = []

# Open files
f_accel = open("C:/Users/Thanasoulkas/Downloads/Accel_output.dat", "rb")
f_gyro = open("C:/Users/Thanasoulkas/Downloads/Gyro_output.dat", "rb")
f_magnet = open("C:/Users/Thanasoulkas/Downloads/Magnet_output.dat", "rb")

### Parse the content of the files and store the data into lists ###

# For the accelerometer data
num = f_accel.read(8)

while num:
    
    num = int.from_bytes(num, byteorder='big')
    accel_time_data.append(num)
    
    row = []
    
    num = f_accel.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    num = f_accel.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    num = f_accel.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    
    accel_data.append(row)
    
    num = f_accel.read(8)

f_accel.close()

# For the gyroscope data
num = f_gyro.read(8)

while num:
    
    num = int.from_bytes(num, byteorder='big')
    gyro_time_data.append(num)
    
    row = []
    
    num = f_gyro.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    num = f_gyro.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    num = f_gyro.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    
    gyro_data.append(row)
    
    num = f_gyro.read(8)

f_gyro.close()

# For the magnetometer data
num = f_magnet.read(8)

while num:
    
    num = int.from_bytes(num, byteorder='big')
    magnet_time_data.append(num)
    
    row = []
    
    num = f_magnet.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    num = f_magnet.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    num = f_magnet.read(4)
    [num] = struct.unpack('>f', num)
    row.append(num)
    
    magnet_data.append(row)
    
    num = f_magnet.read(8)

f_magnet.close()

# Convert the data into DataFrames
df_accel = pd.DataFrame(accel_data, columns = ["X", "Y", "Z"])
df_gyro = pd.DataFrame(gyro_data, columns = ["X", "Y", "Z"])
df_magnet = pd.DataFrame(magnet_data, columns = ["X", "Y", "Z"])

### Pre-process signals ###

# Insert the Timestamp column
df_accel.insert(0, "Timestamp", accel_time_data)
df_gyro.insert(0, "Timestamp", gyro_time_data)
df_magnet.insert(0, "Timestamp", magnet_time_data)

# Drop duplicates based on timestamps
df_accel = df_accel.drop_duplicates(subset = 'Timestamp', keep = 'first').reset_index(drop = True)
df_gyro = df_gyro.drop_duplicates(subset = 'Timestamp', keep = 'first').reset_index(drop = True)
df_magnet = df_magnet.drop_duplicates(subset = 'Timestamp', keep = 'first').reset_index(drop = True)

# Align signals through timestamps
latest_stamp = max(accel_time_data[0], gyro_time_data[0], magnet_time_data[0])

time_diffs = np.array([i - latest_stamp for i in accel_time_data])
first_idx = np.where(time_diffs >= 0)[0][0]
df_accel = df_accel[first_idx :].reset_index(drop = True)

time_diffs = np.array([i - latest_stamp for i in gyro_time_data])
first_idx = np.where(time_diffs >= 0)[0][0]
df_gyro = df_gyro[first_idx :].reset_index(drop = True)

time_diffs = np.array([i - latest_stamp for i in magnet_time_data])
first_idx = np.where(time_diffs >= 0)[0][0]
df_magnet = df_magnet[first_idx :].reset_index(drop = True)

# Remove gravity effect from accelerometer through a high pass filter with 1 Hz cutoff
estimated_fs_acc = round(TIME_CONV / np.mean(np.diff(df_accel['Timestamp'])))
df_accel = filter_gravity_effect(df_accel, 513, CUTOFF, estimated_fs_acc)

# Interpolate values based on evelny aranged time axis for each signal and perform resampling to even out the samples for all signals
df_accel = preproc_signal(df_accel, TIME_CONV, FS)
df_gyro = preproc_signal(df_gyro, TIME_CONV, FS)
df_magnet = preproc_signal(df_magnet, TIME_CONV, FS)

# Keep the axis with the less samples as the global time axis
if((len(df_magnet.Timestamp) <= len(df_gyro.Timestamp)) and ((len(df_magnet.Timestamp)) <= len(df_accel.Timestamp))):
    
    time_axis = df_magnet['Timestamp']
elif((len(df_gyro.Timestamp) <= len(df_magnet.Timestamp)) and ((len(df_gyro.Timestamp)) <= len(df_accel.Timestamp))):
    
    time_axis = df_gyro['Timestamp']
else:
    
    time_axis = df_accel['Timestamp']

# Convert time axis to millis if needed
if(IS_IN_NANOS):
    
    time_axis = time_axis/10**6

# Adjust the timestamps to correspond to video time in ms (based on the spike of Accel_X/Accel_Y/Accel_Z)
idx_of_clap = df_accel['Z'].idxmax()
time_axis = [t - time_axis[idx_of_clap] + TIMESTAMP_OF_CLAP_IN_VIDEO for t in time_axis]

# Keep a Timestamps dataframe from the now synchronized timestamps and display them in human readable form
timestamps = pd.DataFrame(time_axis, columns = ['Timestamp'])
timestamps['Timestamp'] = [str(datetime.timedelta(milliseconds = ms)) for ms in timestamps['Timestamp']]

# Keep a Class dataframe and initially label every move as non-puff (-1)
the_class = pd.DataFrame([-1]*len(time_axis), columns = ['Class'])

# Plot the three signals' figures with timestamps in the x-axis
aranged_samples = np.array([i for i in range(0, len(timestamps), len(timestamps)//10)])
m_xticks = timestamps['Timestamp'][aranged_samples]

plt.figure()

plt.subplot(311)
plt.xticks(aranged_samples, m_xticks)
plt.grid()
plt.plot(df_accel["X"])
plt.plot(df_accel["Y"])
plt.plot(df_accel["Z"])

plt.subplot(312)
plt.xticks(aranged_samples, m_xticks)
plt.grid()
plt.plot(df_gyro["X"])
plt.plot(df_gyro["Y"])
plt.plot(df_gyro["Z"])

plt.subplot(313)
plt.xticks(aranged_samples, m_xticks)
plt.grid()
plt.plot(df_magnet["X"])
plt.plot(df_magnet["Y"])
plt.plot(df_magnet["Z"])

plt.show()

### To label data based on timestamp, use the following command ###
# the_class["Class"][idx_start:idx_stop] = 1
### where idxs are found based on the corresponding timestamps ###

### Save the class series to csv with following commands ###
# f = open('puff_series_X.csv', 'a')
# for row in the_class['Class']:
#   f.write(str(row) + '\n')

### Read the class series into a list with following commands ###
# the_class = []
# import csv
# f = open('puff_series_X.csv', 'r')
# for row in csv.reader(f):
#   the_class.append(int(row[0]))

### To calculate useful session metrics use the following command ###
# sess_metrics = calc_sess_metrics(list(the_class.Class), time_axis)
