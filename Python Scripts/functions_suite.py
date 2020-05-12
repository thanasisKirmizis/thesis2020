
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
# return: the desired features (6)

def extract_features_series(signal, window_len):
    
    import pandas as pd
    
    mean_series = []
    std_series = []
    min_series = []
    max_series = []
    median_series = []
    q_dev_series = []

    for i in range(len(signal) - window_len + 1):
        
        features = extract_features_per_window(signal[i : i+window_len])
        
        mean_series.append(features[0])
        std_series.append(features[1])
        min_series.append(features[2])
        max_series.append(features[3])
        median_series.append(features[4])
        q_dev_series.append(features[5])
        
    df_features = pd.DataFrame(mean_series, columns = ["Mean"])
    df_features.insert(1, "StD", std_series)
    df_features.insert(2, "Min", min_series)
    df_features.insert(3, "Max", max_series)
    df_features.insert(4, "Median", median_series)
    df_features.insert(5, "QuartDev", q_dev_series)
    
    return df_features



### Extracts the features from a given window of samples ###

# window_samples: the samples withing a specific window
# return: the desired features (6) for the given window in dataframe

def extract_features_per_window(window_samples):
    
    import statistics as st
    
    mean = st.mean(window_samples)
    
    std = st.stdev(window_samples)
    
    minimum = min(window_samples)
    
    maximum = max(window_samples)
    
    median = st.median(window_samples)
    
    q1 = st.median(window_samples[:len(window_samples)//2])
    q3 = st.median(window_samples[len(window_samples)//2:])
    quartile_dev = (q3 - q1) / 2
    
    features = [mean, std, minimum, maximum, median, quartile_dev]
    
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