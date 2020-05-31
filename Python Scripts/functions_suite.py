
### Extracts the series of +1 and -1 on a per sliding window basis ###

# signal: the signal to extract from in labeled form (with +1 and -1)
# window_len: the sliding window length in number of samples
# overlap_thres: the overlap's threshold(%) of +1 and -1 values 
#                lower than which the extracted label will be -1
# return: the series of +1 and -1 for the given signal

def extract_puff_series(signal, window_len, overlap_thres):
    
    extracted_series = []
    
    for i in range(len(signal) - window_len + 1):
        
        window_sum = sum(signal[i : i+window_len])
        overlap = window_len + window_sum
        
        if(overlap < overlap_thres*window_len*2):
            
            extracted_series.append(-1)
            
        else:
            
            extracted_series.append(1)
            
    return extracted_series


### Extracts the series of +1 and -1 on a per sliding window basis ###

# signal: the signal to extract from in labeled form (with +1 and -1)
# window_len: the sliding window length in number of samples
# overlap: the overlap (%) between each two adjacent windows
# overlap_thres: the overlap's threshold(%) of +1 and -1 values 
#                lower than which the extracted label will be -1
# return: the series of +1 and -1 for the given signal

def extract_smoke_series(signal, window_len, overlap, overlap_thres):
    
    extracted_series = []
    
    overlap_in_samples = round(window_len*overlap)
    
    for i in range(0, len(signal) - window_len, overlap_in_samples):
        
        window_sum = sum(signal[i : i+window_len])
        overlap_sum = window_len + window_sum
        
        if(overlap_sum < overlap_thres*window_len*2):
            
            extracted_series.append(-1)
            
        else:
            
            extracted_series.append(1)
    
    extracted_series.append(-1)
        
    return extracted_series


### Extracts the series of +1, +2 and -1 on a per sliding window basis ###

# signal: the signal to extract from in labeled form (with +1, +2, and -1)
# window_len: the sliding window length in number of samples
# overlap: the overlap (%) between each two adjacent windows
# return: the series of +1 and -1 for the given signal

def extract_three_class_smoke_series(signal, window_len, overlap):
    
    extracted_series = []
    
    overlap_in_samples = round(window_len*overlap)
    
    for i in range(0, len(signal) - window_len, overlap_in_samples):
        
        window = signal[i : i+window_len]
        class_a = window.count(-1)
        class_b = window.count(1)
        class_c = window.count(2)
        
        if(class_a >= class_b and class_a >= class_c):
            
            extracted_series.append(-1)
            
        elif(class_b >= class_a and class_b >= class_c):
            
            extracted_series.append(1)
            
        elif(class_c >= class_a and class_c >= class_b):
            
            extracted_series.append(2)
    
    extracted_series.append(-1)
        
    return extracted_series


### Extracts only simple features (mean & std) of a signal on a per sliding window basis ###

# signal: the signal to extract the features from
# window_len: the sliding window length in number of samples
# return: the desired features (2)

def extract_simple_features_series(signal, window_len):
    
    import pandas as pd
    import statistics as st
    
    mean_series = []
    std_series = []

    for i in range(len(signal) - window_len + 1):
        
        mean = st.mean(signal[i : i+window_len])
        std = st.stdev(signal[i : i+window_len])
        
        mean_series.append(mean)
        std_series.append(std)
        
    df_features = pd.DataFrame(mean_series, columns = ["Mean"])
    df_features.insert(1, "StD", std_series)
    
    return df_features


### Extracts the features of a signal on a per sliding window basis ###

# signal: the signal to extract the features from
# window_len: the sliding window length in number of samples
# overlap: overlap (%) between windows in each iteration
# return: the desired features (6)

def extract_features_series(signal, window_len, overlap):
    
    import pandas as pd
    
    mean_series = []
    std_series = []
    min_series = []
    max_series = []
    skew_series = []
    kurt_series = []
    
    overlap_in_samples = round(window_len*overlap)

    for i in range(0, len(signal) - window_len, overlap_in_samples):
        
        features = extract_features_per_window(signal[i : i+window_len])
        
        mean_series.append(features[0])
        std_series.append(features[1])
        min_series.append(features[2])
        max_series.append(features[3])
        skew_series.append(features[4])
        kurt_series.append(features[5])
        
    features = extract_features_per_window(signal[i+overlap_in_samples : ])
        
    mean_series.append(features[0])
    std_series.append(features[1])
    min_series.append(features[2])
    max_series.append(features[3])
    skew_series.append(features[4])
    kurt_series.append(features[5])
        
    df_features = pd.DataFrame(mean_series, columns = ["Mean"])
    df_features.insert(1, "StD", std_series)
    df_features.insert(2, "Min", min_series)
    df_features.insert(3, "Max", max_series)
    df_features.insert(4, "Skewness", skew_series)
    df_features.insert(5, "Kurtosis", kurt_series)
    
    return df_features



### Extracts the features from a given window of samples ###

# window_samples: the samples withing a specific window
# return: the desired features (6) for the given window in dataframe

def extract_features_per_window(window_samples):
    
    import statistics as st
    from scipy.stats import kurtosis, skew
    
    mean = st.mean(window_samples)
    
    std = st.stdev(window_samples)
    
    minimum = min(window_samples)
    
    maximum = max(window_samples)
    
    skewness = skew(window_samples)
    
    kurt = kurtosis(window_samples)
    
    features = [mean, std, minimum, maximum, skewness, kurt]
    
    return features



### Extracts the magnitude for all samples of a given 3D signal ###

# signal: a signal with three dimensions (columns)
# return: -1 if signal is not 3D or else its series of magnitudes

def extract_magnitude_dim(signal):
    
    import math
    
    mag = []
    
    if(len(signal[0]) != 3):
        
        print("This only works for three dimensional signals!")
        return -1
    else:
        
        for row in signal:
            
            m = 0
            for i in row:
                
                m = m + pow(i,2)
            
            m = math.sqrt(m)
            mag.append(m)
    
    return mag



### Preprocesses a signal as needed ###

# df_signal: a signal as dataframe with four columns (Timestamp, X, Y, Z)
# time_conv: the factor that converts the time unit of the timestamps into seconds (e.g. 10**9 for nanoseconds)
# target_fs: the desired frequency rate of the processed signal
# return: -1 if signal does not have 4 columns or else the preprocessed signal in dataframe

def preproc_signal(df_signal, time_conv, target_fs):
    
    import pandas as pd
    import numpy as np
    from scipy.signal import resample_poly
    from scipy.interpolate import interp1d
    
    if(len(df_signal.columns) != 4):
        
        print("This only works for four-columned dataframes (Time, X, Y, Z)!")
        return -1
    else:
        
        # Create a time axis based on evenly aranged (in time) samples
        estimated_fs = round(time_conv / np.mean(np.diff(df_signal['Timestamp'])))
        time_axis = np.arange(df_signal['Timestamp'][0], df_signal['Timestamp'][len(df_signal)-1], time_conv/estimated_fs)
        
        # Create the interpolate functions for each dimension based on the values of the signal
        fx = interp1d(df_signal['Timestamp'], df_signal['X'])
        fy = interp1d(df_signal['Timestamp'], df_signal['Y'])
        fz = interp1d(df_signal['Timestamp'], df_signal['Z'])
        
        # Create the new signals that are produced by the interpolation functions based on the new time axis
        x_new = fx(time_axis)
        y_new = fy(time_axis)
        z_new = fz(time_axis)
        
        # Unite the dimensions into one dataframe
        df_signal = pd.DataFrame(data = list(zip(time_axis, x_new, y_new, z_new)), columns = df_signal.columns)
        
        # Perform resampling to even out the number of samples
        df_signal = pd.DataFrame(resample_poly(df_signal, target_fs, estimated_fs), columns = df_signal.columns)
        
        return df_signal



### Filters out the gravity effect for an accelerometer signal ###

# df_accel: a signal as dataframe with three dimensions (columns)
# numtaps: the number of FIR taps (order of filter)
# cutoff: the cutoff frequency in Hz
# fs: the frequency of the signal in Hz
# return: -1 if signal is not 3D or else the filtered signal in dataframe

def filter_gravity_effect(df_accel, numtaps, cutoff, fs):
    
    import pandas as pd
    import numpy as np
    from scipy.signal import firwin
    
    if(len(df_accel.columns) != 4):
        
        print("This only works for three dimensional signals!")
        return -1
    else:
        
        nyq = 2 * cutoff/fs 
        
        sig_x = df_accel.iloc[:,1]
        sig_y = df_accel.iloc[:,2]
        sig_z = df_accel.iloc[:,3]
        
        the_filter = firwin(numtaps, nyq, pass_zero = False)
        
        filt_sig_x = np.convolve(the_filter, sig_x, 'same')
        filt_sig_y = np.convolve(the_filter, sig_y, 'same')
        filt_sig_z = np.convolve(the_filter, sig_z, 'same')
        
        filt_df_accel = pd.DataFrame(np.column_stack((df_accel['Timestamp'], filt_sig_x, filt_sig_y, filt_sig_z)), columns = df_accel.columns)
        
        return filt_df_accel
    


### Calculates some useful metrics of a smoking session ###

# puff_series: a list with the puff series (labeled as +1 or -1) of the smoking session
# timestamps: a list with the time axis that is synchronized with the puff series (must be in milliseconds)
# return: a dataframe containing: 'Session duration', 'No. of Puffs', 'Total duration of Puffs', 'Total duration of non-Puffing',
#                                 'Avg duration of a Puff', 'Std of duration of Puffs', 'Median of duration of Puffs'

def calc_sess_metrics(puff_series, timestamps):
    
    import pandas as pd
    import numpy as np
    
    # Disregard the samples before the beginning of smoking session (negative timestamps)
    timestamps = [t for t in timestamps if t>=0]
    puff_series = puff_series[len(puff_series) - len(timestamps) : ]
    
    # Useful variables definitions
    puff_times = []
    start_t = 0
    end_t = 0
    
    # Loop through the Puff Series
    for i in range(len(puff_series)-1):
        
        # If located the start of a puff
        if( (puff_series[i] == -1) and (puff_series[i+1] == 1) ):
            
            # Keep the starting time of this puff
            start_t = timestamps[i+1]
            
        # If located the ending of a puff
        if( (puff_series[i] == 1) and (puff_series[i+1] == -1) ):
            
            # Keep the ending time of this puff
            end_t = timestamps[i+1]
            
            # Store the duration to the list
            puff_times.append(end_t - start_t)
            
    # Calculate the various desired metrics
    sess_dur = timestamps[-1] - timestamps[0]
    num_puffs = len(puff_times)
    tot_dur_puff = sum(puff_times)
    tot_dur_non_puff = sess_dur - tot_dur_puff
    avg_dur = np.mean(puff_times)
    std_dur = np.std(puff_times)
    med_dur = np.median(puff_times)
    
    # Create dataframe and return
    labels = ['SessDur', 'NumPuffs', 'TotDurPuff', 'TotDurNonPuff', 'AvgPuffDur', 'StdPuffDur', 'MedianPuffDur']
    sess_metrics = pd.DataFrame([[sess_dur, num_puffs, tot_dur_puff, tot_dur_non_puff, avg_dur, std_dur, med_dur]], columns = labels)
    
    return sess_metrics



### Extracts the f1 score for the sum of confusion matrices of all participants ###

# conf_matrices: the list of confusion matrices for all participants
# return: the final f1 score

def extract_final_score(conf_matrices):
    
    # Outer zero-padding like this:
    # a = np.pad(a, [(0,1), (0, 1)], mode='constant')
    
    return final_score