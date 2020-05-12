
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import struct

# Definitions
TIMESTAMP_OF_CLAP_IN_VIDEO = 8000

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

# Subsample the accelerometer's data with 1/2 frequency
accel_time_data = accel_time_data[::2]
accel_data = accel_data[::2]

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

# Convert the data into DataFrames with everything initially labeled as non-puff (-1)
df_accel = pd.DataFrame(accel_data, columns = ["X", "Y", "Z"])
df_accel.insert(0, "Timestamp", accel_time_data)
df_accel.insert(4, "Class", [-1]*len(accel_data))
 
df_gyro = pd.DataFrame(gyro_data, columns = ["X", "Y", "Z"])
df_gyro.insert(0, "Timestamp", gyro_time_data)
df_gyro.insert(4, "Class", [-1]*len(gyro_data))

df_magnet = pd.DataFrame(magnet_data, columns = ["X", "Y", "Z"])
df_magnet.insert(0, "Timestamp", magnet_time_data)
df_magnet.insert(4, "Class", [-1]*len(magnet_data))

# Adjust the timestamps to correspond to video time in ms (based on the spike of Accel_X)
idx_of_clap = df_accel['X'].idxmax()

df_accel['Timestamp'] = df_accel['Timestamp'] - df_accel['Timestamp'][idx_of_clap] + TIMESTAMP_OF_CLAP_IN_VIDEO
df_gyro['Timestamp'] = df_gyro['Timestamp'] - df_gyro['Timestamp'][idx_of_clap] + TIMESTAMP_OF_CLAP_IN_VIDEO
df_magnet['Timestamp'] = df_magnet['Timestamp'] - df_magnet['Timestamp'][idx_of_clap] + TIMESTAMP_OF_CLAP_IN_VIDEO

# Plot the three signals' figures
plt.figure()

plt.subplot(311)
plt.plot(df_accel["X"])
plt.plot(df_accel["Y"])
plt.plot(df_accel["Z"])

plt.subplot(312)
plt.plot(df_gyro["X"])
plt.plot(df_gyro["Y"])
plt.plot(df_gyro["Z"])

plt.subplot(313)
plt.plot(df_magnet["X"])
plt.plot(df_magnet["Y"])
plt.plot(df_magnet["Z"])

plt.show()

### To label data based on timestamp, use the following command ###
# df_accel["Class"][idx_start:idx_stop] = 1
# where idxs are found based on the corresponding timestamps
