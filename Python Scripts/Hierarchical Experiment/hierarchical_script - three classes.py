
# Imports
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import *
from functions_suite import extract_features_series, extract_magnitude_dim, extract_three_class_smoke_series

# Definitions
window_secs = 15
sample_rate = 50
window_len_samples = window_secs * sample_rate
overlap = 0.5

### Load data from files and create a data frame for each participant ###
participants_dfs = []
participants_classes = []

# Define directory name
folder = 'J:/Smoking Datasets/UT_Smoking_Data/csvs_full'

# For each participant
for file in os.listdir(folder):
    
    print(">>>>> Reading file: ", file)
    
    # Define csv file name to be opened
    filename = folder + '/' + file
    
    # Name the columns of the dataframe
    fields = ["Accel_X", "Accel_Y", "Accel_Z",
              "Gyro_X", "Gyro_Y", "Gyro_Z",
              "Magnet_X", "Magnet_Y", "Magnet_Z"] 
    rows = []
    classes = []
    
    # Open and read the csv file
    f = open(filename, "r")
    csvreader = csv.reader(f)
    
    ## Extract each data row one by one and read data as floats (and classes as strings)
    for row in csvreader: 
        
        # Store the desired data
        rows.append([float(i) for i in row[1:13] if i not in row[4:7]])
        classes.append(row[31])
    
    # Convert the data into dataframe 
    df = pd.DataFrame(rows, columns = fields)
    classes = pd.DataFrame(classes, columns = ["Activity"])
    
#    ### KEEP 20,000 SAMPLES OF EACH CLASS FOR DEMO ###
#    sss = StratifiedShuffleSplit(n_splits = 1, train_size = 20000)
#    for train_index, test_index in sss.split(df, classes):
#        
#        df = df.loc[train_index].reset_index(drop = True)
#        classes = classes.loc[train_index].reset_index(drop = True)
    
    print(">> Adding Magnitude Dimension...")
    
    ## Add the 'Magnitude' dimension for each signal of the data
    accel_signal = np.transpose([list(df["Accel_X"]), list(df["Accel_Y"]), list(df["Accel_Z"])])
    np.nan_to_num(accel_signal, copy = False)
    mag = extract_magnitude_dim(accel_signal)
    df.insert(3, 'Accel_Mag', mag)
      
    gyro_signal = np.transpose([list(df["Gyro_X"]), list(df["Gyro_Y"]), list(df["Gyro_Z"])])
    np.nan_to_num(gyro_signal, copy = False)
    mag = extract_magnitude_dim(gyro_signal)
    df.insert(7, 'Gyro_Mag', mag)
      
    magnet_signal = np.transpose([list(df["Magnet_X"]), list(df["Magnet_Y"]), list(df["Magnet_Z"])])
    np.nan_to_num(magnet_signal, copy = False)
    mag = extract_magnitude_dim(magnet_signal)
    df.insert(11, 'Magnet_Mag', mag)
    
    np.nan_to_num(df, copy = False)
    
    print(">> Extracting Features...")
    
    # Extract the features series for each dimension of each signal (and apply some renaming)
    accel_x_features = extract_features_series(df["Accel_X"], window_len_samples, overlap)
    accel_x_features.columns = ['AccX' + str(col) for col in accel_x_features.columns]
    accel_y_features = extract_features_series(df["Accel_Y"], window_len_samples, overlap)
    accel_y_features.columns = ['AccY' + str(col) for col in accel_y_features.columns]
    accel_z_features = extract_features_series(df["Accel_Z"], window_len_samples, overlap)
    accel_z_features.columns = ['AccZ' + str(col) for col in accel_z_features.columns]
    accel_mag_features = extract_features_series(df["Accel_Mag"], window_len_samples, overlap)
    accel_mag_features.columns = ['AccMag' + str(col) for col in accel_mag_features.columns]
      
    gyro_x_features = extract_features_series(df["Gyro_X"], window_len_samples, overlap)
    gyro_x_features.columns = ['GyroX' + str(col) for col in gyro_x_features.columns]
    gyro_y_features = extract_features_series(df["Gyro_Y"], window_len_samples, overlap)
    gyro_y_features.columns = ['GyroY' + str(col) for col in gyro_y_features.columns]
    gyro_z_features = extract_features_series(df["Gyro_Z"], window_len_samples, overlap)
    gyro_z_features.columns = ['GyroZ' + str(col) for col in gyro_z_features.columns]
    gyro_mag_features = extract_features_series(df["Gyro_Mag"], window_len_samples, overlap)
    gyro_mag_features.columns = ['GyroMag' + str(col) for col in gyro_mag_features.columns]
      
    magnet_x_features = extract_features_series(df["Magnet_X"], window_len_samples, overlap)
    magnet_x_features.columns = ['MagnetX' + str(col) for col in magnet_x_features.columns]
    magnet_y_features = extract_features_series(df["Magnet_Y"], window_len_samples, overlap)
    magnet_y_features.columns = ['MagnetY' + str(col) for col in magnet_y_features.columns]
    magnet_z_features = extract_features_series(df["Magnet_Z"], window_len_samples, overlap)
    magnet_z_features.columns = ['MagnetZ' + str(col) for col in magnet_z_features.columns]
    magnet_mag_features = extract_features_series(df["Magnet_Mag"], window_len_samples, overlap)
    magnet_mag_features.columns = ['MagnetMag' + str(col) for col in magnet_mag_features.columns]
    
    # Create the feature set (per columns)
    desired_features = pd.concat([accel_x_features, accel_y_features, accel_z_features, accel_mag_features,
                                  gyro_x_features, gyro_y_features, gyro_z_features, gyro_mag_features,
                                  magnet_x_features, magnet_y_features, magnet_z_features, magnet_mag_features],
                                 axis = 1)
    
    print(">> Normalizing Dataset...")

    # Normalize the dataset
    val = desired_features.values
    min_max_scaler = preprocessing.MinMaxScaler()
    val_scaled = min_max_scaler.fit_transform(val)
    desired_features = pd.DataFrame(val_scaled, columns=desired_features.columns)
    
    # Add the features data frame to the list of participant's data frames
    participants_dfs.append(desired_features)
    
    print(">> Adjusting Classes...")
    
    # Name the classes as "Smoke" or "Non-Smoke"
    classes.loc[np.where(classes["Activity"] == "SmokeSD")] = 1
    classes.loc[np.where(classes["Activity"] == "SmokeST")] = 1
    classes.loc[np.where(classes["Activity"] == "SmokeWK")] = 2
    classes.loc[np.where(classes["Activity"] == "SmokeGP")] = 1
    classes.loc[np.where((classes["Activity"] != 1) & (classes["Activity"] != 2))] = -1
    
    # Adjust the class column to the results of the sliding window
    classes = pd.DataFrame(extract_three_class_smoke_series(list(classes.Activity), window_len_samples, overlap),
                           columns = ["Activity"])

    # Add the class data frame to the list of participant's data frames
    participants_classes.append(classes)


### Perform Leave-one-subject-out cross validation and keep confusion matrices ###
    
conf_mat_list = []
acc_list = []
f1_list = []
recall_list = []
precision_list = []

for i in range(len(participants_dfs)):
    
    print(">>>>> Performing LOSO #", i)
    
    train_subjects = [k for j,k in enumerate(participants_dfs) if j!=i]
    train_subjects = pd.concat(train_subjects)
    train_classes = [k for j,k in enumerate(participants_classes) if j!=i]
    train_classes = pd.concat(train_classes)
    
    test_subject = participants_dfs[i]
    test_class = participants_classes[i]
    
    rfc = RandomForestClassifier(random_state = 1, class_weight = 'balanced', n_estimators = 9)
    rfc.fit(train_subjects, train_classes.values.ravel())
    
    class_pred = rfc.predict(test_subject)

    conf_mat_list.append(confusion_matrix(test_class, class_pred))
    acc_list.append(accuracy_score(test_class, class_pred))
    f1_list.append(f1_score(test_class, class_pred, average='micro'))
    recall_list.append(recall_score(test_class, class_pred, average='micro'))
    precision_list.append(precision_score(test_class, class_pred, average='micro'))

