import pickle
import sys

import numpy as np
from scipy import signal
from scipy.stats import entropy
import pywt
import pandas as pd
from sklearn.model_selection import train_test_split

# logging
sys.stderr = open(snakemake.log[0], 'w')


def cal_mean_crossings(sig: pd.Series):
    """
    Calculates the feature mean crossings for a given pandas series
    :param sig: A pandas series object
    :return: The number of times the signal crosses the mean.
    """
    return len(np.where(np.diff(np.signbit(sig - np.mean(sig))))[0])


def cal_histogram_features(sig: pd.Series):
    """
    Calculates the histogram featues from a given signal
    :param sig: A pandas Series object that contains accelemeter data
    :return:
    """
    # Divide the data in 4 bins
    histo = np.histogram(sig, bins=4, density=True)[0]
    return histo


def extract_features(segments: list, sampling_frequency: float) -> pd.DataFrame:
    """
    Calculates features from accelerometer data based on segments
    :param sampling_frequency: The chosen resampled frequency
    :param segments: A list containing segmented accelerometer data
    :return: A pandas dataframe that contains the obtained features
    """
    # initialize dictionary with features
    features = {'Movement': [],
                'Mean_x': [], 'Mean_y': [], 'Mean_z': [], 'Mean_comb': [],
                'Median_x': [], 'Median_y': [], 'Median_z': [], 'Median_comb': [],
                'StD_x': [], 'StD_y': [], 'StD_z': [], 'StD_comb': [],
                'RMS_x': [], 'RMS_y': [], 'RMS_z': [], 'RMS_comb': [],
                'Variance_x': [], 'Variance_y': [], 'Variance_z': [], 'Variance_comb': [],
                'Energy_x': [], 'Energy_y': [], 'Energy_z': [], 'Energy_comb': [],
                'Entropy_x': [], 'Entropy_y': [], 'Entropy_z': [], 'Entropy_comb': [],
                'Mean_Frequency_x': [], 'Mean_Frequency_y': [], 'Mean_Frequency_z': [], 'Mean_Frequency_comb': [],
                'Jerk_mean_x': [], 'Jerk_mean_y': [], 'Jerk_mean_z': [], 'Jerk_mean_comb': [],
                'Mean_absolute_deviation_x': [], 'Mean_absolute_deviation_y': [], 'Mean_absolute_deviation_z': [],
                'Mean_absolute_deviation_comb': [],
                'DWT_vm2_x': [], 'DWT_vm2_y': [], 'DWT_vm2_z': [], 'DWT_vm2_comb': [],
                'Average_difference_x': [], 'Average_difference_y': [], 'Average_difference_z': [],
                'Average_difference_comb': [],
                'Correlation_x_y': [], 'Correlation_x_z': [], 'Correlation_y_z': [],
                'Mean_crossings_x': [], 'Mean_crossings_y': [], 'Mean_crossings_z': [], 'Mean_crossings_comb': [],
                '1_hist_x': [], '2_hist_x': [], '3_hist_x': [], '4_hist_x': [],
                '1_hist_y': [], '2_hist_y': [], '3_hist_y': [], '4_hist_y': [],
                '1_hist_z': [], '2_hist_z': [], '3_hist_z': [], '4_hist_z': [],
                '1_hist_comb': [], '2_hist_comb': [], '3_hist_comb': [], '4_hist_comb': [],
                'kurtosis_x': [], 'kurtosis_y': [], 'kurtosis_z': [], 'kurtosis_comb': [],
                'skewness_x': [], 'skewness_y': [], 'skewness_z': [], 'skewness_comb': []}

    # loop through the segments and extract features
    for segment in segments:
        # Add most common movement value of segment
        features['Movement'].append(segment['gt'].mode()[0])
        # Calculate the mean
        features['Mean_x'].append(segment['x'].mean())
        features['Mean_y'].append(segment['y'].mean())
        features['Mean_z'].append(segment['z'].mean())
        features['Mean_comb'].append(segment['comb'].mean())
        # Calculate the median
        features['Median_x'].append(segment['x'].median())
        features['Median_y'].append(segment['y'].median())
        features['Median_z'].append(segment['z'].median())
        features['Median_comb'].append(segment['comb'].median())
        # Calculate the standard deviation
        features['StD_x'].append(segment['x'].std())
        features['StD_y'].append(segment['y'].std())
        features['StD_z'].append(segment['z'].std())
        features['StD_comb'].append(segment['comb'].std())
        # Calculate the RMS
        features['RMS_x'].append(np.sqrt(np.mean(segment['x'] ** 2)))
        features['RMS_y'].append(np.sqrt(np.mean(segment['y'] ** 2)))
        features['RMS_z'].append(np.sqrt(np.mean(segment['z'] ** 2)))
        features['RMS_comb'].append(np.sqrt(np.mean(segment['comb'] ** 2)))
        # Calculate the variance
        features['Variance_x'].append(segment['x'].var())
        features['Variance_y'].append(segment['y'].var())
        features['Variance_z'].append(segment['x'].var())
        features['Variance_comb'].append(segment['comb'].var())
        # Calculate the jerk
        features['Jerk_mean_x'].append(np.mean(np.diff(segment['x']) / np.diff(segment['Frequency_Time'])))
        features['Jerk_mean_y'].append(np.mean(np.diff(segment['y']) / np.diff(segment['Frequency_Time'])))
        features['Jerk_mean_z'].append(np.mean(np.diff(segment['z']) / np.diff(segment['Frequency_Time'])))
        features['Jerk_mean_comb'].append(np.mean(np.diff(segment['comb']) / np.diff(segment['Frequency_Time'])))
        # Mean absolute deviation
        features['Mean_absolute_deviation_x'].append(segment['x'].mad())
        features['Mean_absolute_deviation_y'].append(segment['y'].mad())
        features['Mean_absolute_deviation_z'].append(segment['z'].mad())
        features['Mean_absolute_deviation_comb'].append(segment['comb'].mad())

        # Correlation between axis
        features['Correlation_x_y'].append(segment['x'].corr(segment['y']))
        features['Correlation_x_z'].append(segment['x'].corr(segment['z']))
        features['Correlation_y_z'].append(segment['y'].corr(segment['z']))

        # Mean crossings
        features['Mean_crossings_x'].append(cal_mean_crossings(segment['x']))
        features['Mean_crossings_y'].append(cal_mean_crossings(segment['y']))
        features['Mean_crossings_z'].append(cal_mean_crossings(segment['z']))
        features['Mean_crossings_comb'].append(cal_mean_crossings(segment['comb']))

        # Average difference axis
        features['Average_difference_x'].append(np.mean(np.diff(segment['x'])))
        features['Average_difference_y'].append(np.mean(np.diff(segment['y'])))
        features['Average_difference_z'].append(np.mean(np.diff(segment['z'])))
        features['Average_difference_comb'].append(np.mean(np.diff(segment['comb'])))

        # histogram bins
        hist_x = cal_histogram_features(segment['x'])
        features['1_hist_x'].append(hist_x[0])
        features['2_hist_x'].append(hist_x[1])
        features['3_hist_x'].append(hist_x[2])
        features['4_hist_x'].append(hist_x[3])

        hist_y = cal_histogram_features(segment['y'])
        features['1_hist_y'].append(hist_y[0])
        features['2_hist_y'].append(hist_y[1])
        features['3_hist_y'].append(hist_y[2])
        features['4_hist_y'].append(hist_y[3])

        hist_z = cal_histogram_features(segment['z'])
        features['1_hist_z'].append(hist_z[0])
        features['2_hist_z'].append(hist_z[1])
        features['3_hist_z'].append(hist_z[2])
        features['4_hist_z'].append(hist_z[3])

        hist_comb = cal_histogram_features(segment['comb'])
        features['1_hist_comb'].append(hist_comb[0])
        features['2_hist_comb'].append(hist_comb[1])
        features['3_hist_comb'].append(hist_comb[2])
        features['4_hist_comb'].append(hist_comb[3])

        # kurtosis feature calculation
        kurtosis_x = segment['x'].kurtosis()
        features['kurtosis_x'].append(kurtosis_x)

        kurtosis_y = segment['y'].kurtosis()
        features['kurtosis_y'].append(kurtosis_y)

        kurtosis_z = segment['z'].kurtosis()
        features['kurtosis_z'].append(kurtosis_z)

        kurtosis_comb = segment['comb'].kurtosis()
        features['kurtosis_comb'].append(kurtosis_comb)

        # Skewness
        skewness_x = segment['x'].skew()
        features['skewness_x'].append(skewness_x)

        skewness_y = segment['y'].skew()
        features['skewness_y'].append(skewness_y)

        skewness_z = segment['z'].skew()
        features['skewness_z'].append(skewness_z)

        skewness_comb = segment['comb'].skew()
        features['skewness_comb'].append(skewness_comb)

        # Extract features from frequency domain
        # Using fourier transform
        f_comb, pxx_den_comb = signal.periodogram(segment['comb'], sampling_frequency)

        features['Mean_Frequency_comb'].append(np.mean(pxx_den_comb))
        features['Energy_comb'].append(np.sum(pxx_den_comb))

        f_x, pxx_den_x = signal.periodogram(segment['x'], sampling_frequency)
        features['Mean_Frequency_x'].append(np.mean(pxx_den_x))
        features['Energy_x'].append(np.sum(pxx_den_x))

        f_y, pxx_den_y = signal.periodogram(segment['y'], sampling_frequency)
        features['Mean_Frequency_y'].append(np.mean(pxx_den_y))
        features['Energy_y'].append(np.sum(pxx_den_y))

        f_z, pxx_den_z = signal.periodogram(segment['z'], sampling_frequency)
        features['Mean_Frequency_z'].append(np.mean(pxx_den_z))
        features['Energy_z'].append(np.sum(pxx_den_z))

        # Using DWT
        # Use DWT till lvl 8
        data_comb = segment['comb']
        data_x = segment['x']
        data_y = segment['y']
        data_z = segment['z']
        sum_squere_coeff = 0
        sum_squere_coeff_x = 0
        sum_squere_coeff_y = 0
        sum_squere_coeff_z = 0
        for i in range(8):
            (data_comb, coeff_d) = pywt.dwt(segment['comb'], 'DB10')
            sum_squere_coeff += sum(i * i for i in coeff_d)

            (data_x, coeff_d_x) = pywt.dwt(data_x, 'DB10')
            sum_squere_coeff_x += sum(i * i for i in coeff_d_x)

            (data_y, coeff_d_y) = pywt.dwt(data_y, 'DB10')
            sum_squere_coeff_y += sum(i * i for i in coeff_d_y)

            (data, coeff_d_z) = pywt.dwt(data_z, 'DB10')
            sum_squere_coeff_z += sum(i * i for i in coeff_d_z)

        features['DWT_vm2_x'].append(sum_squere_coeff_x / sum(i * i for i in segment['x']))
        features['DWT_vm2_y'].append(sum_squere_coeff_y / sum(i * i for i in segment['y']))
        features['DWT_vm2_z'].append(sum_squere_coeff_z / sum(i * i for i in segment['z']))
        features['DWT_vm2_comb'].append(sum_squere_coeff / sum(i * i for i in segment['comb']))

        # Normalize psd
        pxx_den_norm = pxx_den_comb / np.sum(pxx_den_comb)
        features['Entropy_comb'].append(entropy(pxx_den_norm))
        pxx_den_norm_x = pxx_den_x / np.sum(pxx_den_x)
        features['Entropy_x'].append(entropy(pxx_den_norm_x))
        pxx_den_norm_y = pxx_den_y / np.sum(pxx_den_y)
        features['Entropy_y'].append(entropy(pxx_den_norm_y))
        pxx_den_norm_z = pxx_den_z / np.sum(pxx_den_z)
        features['Entropy_z'].append(entropy(pxx_den_norm_z))
    return pd.DataFrame(features)


def main():
    # read in the segments file and extract the segments list
    with open(snakemake.input[0], 'rb') as input_file:
        segments = pickle.load(input_file)

    # get the chosen frequency
    chosen_freq = snakemake.params['resampled_frequency']

    # extract features
    features = extract_features(segments, chosen_freq)
    features = features.drop(features.loc[features['Movement'] == '0'].index)

    # split into labels and features
    labels = features['Movement']
    features = features.drop(columns=['Movement'])

    # Split in train and test set
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    # write obtained features and labels to the output files
    features_train.to_csv(snakemake.output['features_train'], index=False)
    features_test.to_csv(snakemake.output['features_test'], index=False)
    labels_train.to_csv(snakemake.output['labels_train'], index=False)
    labels_test.to_csv(snakemake.output['labels_test'], index=False)
    return


if __name__ == '__main__':
    sys.exit(main())
