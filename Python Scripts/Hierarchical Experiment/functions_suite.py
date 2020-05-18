
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