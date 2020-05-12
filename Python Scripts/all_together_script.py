
# Imports
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import *
from functions_suite import extract_simple_features_series, extract_magnitude_dim

# Definitions
window_secs = 5
sample_rate = 50
window_len_samples = window_secs * sample_rate

### Load data from file and create a big data frame ###

# Define csv file name to be opened
filename = "C:/Users/Thanasoulkas/Desktop/smartphoneatwrist.csv"

# Name the columns of the dataframe
fields = ["Time Stamp", "Accel_X", "Accel_Y", "Accel_Z",
          "L_Accel_X", "L_Accel_Y", "L_Accel_Z",
          "Gyro_X", "Gyro_Y", "Gyro_Z",
          "Magnet_X", "Magnet_Y", "Magnet_Z",
          "Activity"] 
rows = []

# Open and read the csv file
f = open(filename, "r")
csvreader = csv.reader(f)

# Extract each data row one by one and read data as floats
for row in csvreader: 
    
    # Keep only the walk/stand/coffee/smoke/talk/eat ones
    if(row[13] == "11111" or row[13] == "11112" or row[13] == "11122"
       or row[13] == "11120" or row[13] == "11121" or row[13] == "11123"):
        rows.append([float(i) for i in row])

# Convert the data into dataframe 
df = pd.DataFrame(rows, columns = fields)

# ### KEEP 15,000 SAMPLES OF EACH CLASS FOR TESTING ###
df_walk = df[0:15000]
df_stand = df[90000:159000]
df_coffee = df[180000:195000]
df_talk = df[270000:285000]
df_smoke = df[360000:375000]
df_eat = df[450000:465000]
df = pd.concat([df_walk, df_stand, df_coffee, df_talk, df_smoke, df_eat])

# Add the 'Magnitude' dimension for each signal of the data
accel_signal = np.transpose([list(df["Accel_X"]), list(df["Accel_Y"]), list(df["Accel_Z"])])
mag = extract_magnitude_dim(accel_signal)
df.insert(4, 'Accel_Mag', mag)
  
gyro_signal = np.transpose([list(df["Gyro_X"]), list(df["Gyro_Y"]), list(df["Gyro_Z"])])
mag = extract_magnitude_dim(gyro_signal)
df.insert(8, 'Gyro_Mag', mag)
  
magnet_signal = np.transpose([list(df["Magnet_X"]), list(df["Magnet_Y"]), list(df["Magnet_Z"])])
mag = extract_magnitude_dim(magnet_signal)
df.insert(12, 'Magnet_Mag', mag)
  
### FOR NOW LET'S ONLY KEEP THE MEAN AND STD ###

# Extract the features series for each dimension of each signal
accel_x_features = extract_simple_features_series(df["Accel_X"], window_len_samples)
accel_y_features = extract_simple_features_series(df["Accel_Y"], window_len_samples)
accel_z_features = extract_simple_features_series(df["Accel_Z"], window_len_samples)
accel_mag_features = extract_simple_features_series(df["Accel_Mag"], window_len_samples)
  
gyro_x_features = extract_simple_features_series(df["Gyro_X"], window_len_samples)
gyro_y_features = extract_simple_features_series(df["Gyro_Y"], window_len_samples)
gyro_z_features = extract_simple_features_series(df["Gyro_Z"], window_len_samples)
gyro_mag_features = extract_simple_features_series(df["Gyro_Mag"], window_len_samples)
  
magnet_x_features = extract_simple_features_series(df["Magnet_X"], window_len_samples)
magnet_y_features = extract_simple_features_series(df["Magnet_Y"], window_len_samples)
magnet_z_features = extract_simple_features_series(df["Magnet_Z"], window_len_samples)
magnet_mag_features = extract_simple_features_series(df["Magnet_Mag"], window_len_samples)

# Create the feature set
desired_features = pd.DataFrame(list(accel_x_features["Mean"]), columns = ["AccMeanX"])
desired_features.insert(1, "AccStDX", accel_x_features["StD"])
desired_features.insert(2, "AccMeanY", accel_y_features["Mean"])
desired_features.insert(3, "AccStDY", accel_y_features["StD"])
desired_features.insert(4, "AccMeanZ", accel_z_features["Mean"])
desired_features.insert(5, "AccStDZ", accel_z_features["StD"])
desired_features.insert(6, "AccMeanMag", accel_mag_features["Mean"])
desired_features.insert(7, "AccStDMag", accel_mag_features["StD"])
desired_features.insert(8, "GyroMeanX", gyro_x_features["Mean"])
desired_features.insert(9, "GyroStDX", gyro_x_features["StD"])
desired_features.insert(10, "GyroMeanY", gyro_y_features["Mean"])
desired_features.insert(11, "GyroStDY", gyro_y_features["StD"])
desired_features.insert(12, "GyroMeanZ", gyro_z_features["Mean"])
desired_features.insert(13, "GyroStDZ", gyro_z_features["StD"])
desired_features.insert(14, "GyroMeanMag", gyro_mag_features["Mean"])
desired_features.insert(15, "GyroStDMag", gyro_mag_features["StD"])
desired_features.insert(16, "MagnetMeanX", magnet_x_features["Mean"])
desired_features.insert(17, "MagnetStDX", magnet_x_features["StD"])
desired_features.insert(18, "MagnetMeanY", magnet_y_features["Mean"])
desired_features.insert(19, "MagnetStDY", magnet_y_features["StD"])
desired_features.insert(20, "MagnetMeanZ", magnet_z_features["Mean"])
desired_features.insert(21, "MagnetStDZ", magnet_z_features["StD"])
desired_features.insert(22, "MagnetMeanMag", magnet_mag_features["Mean"])
desired_features.insert(23, "MagnetStDMag", magnet_mag_features["StD"])

# Normalize the dataset
val = desired_features.values
min_max_scaler = preprocessing.MinMaxScaler()
val_scaled = min_max_scaler.fit_transform(val)
desired_features = pd.DataFrame(val_scaled, columns=desired_features.columns)

# Create the class set
classes = df["Activity"][0:-window_len_samples+1].reset_index(drop = True)

# Perform 10-fold Stratified Cross-Validation and store the results in tables
acc_list = []
conf_mat_list = []

skf = StratifiedKFold(n_splits=10)
gnb = GaussianNB()

for train_index, test_index in skf.split(desired_features, classes):
    
    X_train, X_test = desired_features.loc[train_index], desired_features.loc[test_index]
    y_train, y_test = classes.loc[train_index], classes.loc[test_index]

    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    conf_mat_list.append(confusion_matrix(y_test, y_pred))
    acc_list.append(accuracy_score(y_test, y_pred))

