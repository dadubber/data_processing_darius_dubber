import pickle
import sys

import pandas as pd
import numpy as np

# logging
sys.stderr = open(snakemake.log[0], 'w')


def segment_the_data(data: pd.DataFrame, window_size: int, overlap: int) -> list:
    """
    Segments accelerometer data and creates segments of a given length and overlap
    :param data: a pandas dataframe containing accelerometer data
    :param window_size: The length of the segment in seconds
    :param overlap: The overlap of the created segments
    :return: A list containing the created segments
    """
    # Initialize the segments list
    segments = []
    i = 0
    # Start a loop to go through the pandas dataframe
    while True:
        # If the last datapoint is in segments break the loop
        if segments:
            if data.iloc[-1]['Frequency_Time'] in segments[-1].values:
                break
        data = data.reset_index(drop=True)
        # Create a segment
        start = data['Frequency_Time'].iloc[0]
        end_window = data[data['Frequency_Time'] <= (start + window_size)].tail(1).index[0]
        new_window = data.loc[:end_window]
        segments.append(new_window)
        new_start_index = round(len(new_window) / 100 * overlap)
        data = data.loc[new_start_index:]
        i += 1
    return segments


def cal_time_from_freq(time_points: list, sample_freq: int) -> list:
    """
    Calculates time points based on the sampling frequencies
    :param time_points: a list containing the originaacc_user['comb'] = combine_axes(acc_dict[device][user])l timepoints
    :param sample_freq: The sampling frequency of the device
    :return: A list containing the new time points
    """
    seconds_point = 1 / sample_freq
    new_time_points = [(i + 1) * seconds_point for i in range(len(time_points))]
    return new_time_points


def combine_axes(acc_user: pd.DataFrame):
    """
       Axes are combined into a single vector with the equation
       :param df:
       :return:
       """
    comb = np.sqrt(acc_user['x'] ** 2 + acc_user['y'] ** 2 + acc_user['z'] ** 2)
    return comb


def resampling(data: pd.DataFrame, org_freq: float, wanted_freq: float) -> pd.DataFrame:
    """
    function to downsample a dataframe
    cannot be used for upsampling
    :param data:
    :param org_freq:
    :param wanted_freq:
    :return:
    """
    # calculate new step size
    step_size = round(org_freq / wanted_freq)
    return data.iloc[::step_size]


def preprocess_the_data(acc_df: pd.DataFrame, sampling_frequencies: dict, resampled_frequency: float) -> list:
    """
    Function for preprocessing accelerometer data.
    It will remove unnecessarily columns and will segment the data.
    :param resampled_frequency: The frequency that will be used to resample the data
    :param sampling_frequencies: A dictionary containing the real sampling frequencies
    :param acc_df: A pandas dataframe containing the accelerometer data
    :return:
    """
    # Remove arrival time from acc_data
    acc_df = acc_df.drop(['Arrival_Time'], axis=1)
    segments = []

    for device in set(acc_df['Device']):
        # Get all the data for the specific device
        acc_device = acc_df[acc_df['Device'] == device]

        # Remove device column
        acc_device = acc_device.drop(['Device'], axis=1)

        for user in set(acc_device['User']):
            # Get the data of the user
            acc_user = acc_device[acc_device['User'] == user]
            acc_user = acc_user.drop('User', axis=1)

            # Reset index
            acc_user = acc_user.reset_index(drop=True)

            # Get walk, sit, stand, stairsup and stairsdown in the data set
            last_idx_acc = acc_user.where(acc_user['gt'].isin(['sit', 'stand', 'walk', 'stairsup', 'stairsdown'])
                                          ).last_valid_index()

            if last_idx_acc:
                acc_user = acc_user[:last_idx_acc]

            # Add a combination of the axes
            acc_user['comb'] = combine_axes(acc_user)

            # Create Frequency_Time
            acc_user['Frequency_Time'] = cal_time_from_freq(acc_user['Creation_Time'], sampling_frequencies[device])

            # Resample the data with a new frequency
            acc_user = resampling(acc_user, sampling_frequencies[device], resampled_frequency)

            acc_user['gt'] = acc_user['gt'].replace(np.nan, '0')

            # Segment the data
            segments = segment_the_data(acc_user, window_size=2, overlap=50)

    return segments


def main():
    # Load in the dataframe
    with open(snakemake.input[0], 'rb') as input_file:
        acc_df = pickle.load(input_file)

    # get the necessarily parameters
    sampling_frequencies = snakemake.params['sampling_frequencies']
    resampled_frequency = snakemake.params['resampled_frequency']

    # Run the preprocess function
    segments = preprocess_the_data(acc_df, sampling_frequencies, resampled_frequency)

    # Write the results to the output file
    with open(snakemake.output[0], 'wb') as output_file:
        pickle.dump(segments, output_file)


if __name__ == '__main__':
    sys.exit(main())


