from curses import raw
import os
import sys
import time
import csv
import torch
import random
from black import out
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool, cpu_count

"""
NOTE:
    1) Time in milliseconds.
    2) Input file data: [time, label, sensor(3) * pos(5) * dim(3)]
    3) SENSORS = ["acc", "gyr", "mag"]
    4) POSITIONS = ["head", "chest", "upperarm", "waist", "shin"]
    5) Data order: (sensor, pos, dim)
"""

SEGMENT_SPAN = 5000
INTERVAL_SPAN = 1000
SEGMENT_SAMPLE_COUNT = 250
INTERVAL_SAMPLE_COUNT = 50
SEGMENT_INTERVAL_COUNT = 9
OVERLAP_RATIO = 0.5
BODY_POSITION = 5
SENSOR_TYPE = 4
SENSOR_DIM = 3

# Class encodings
ACTIVITIES = ["climbingdown", "climbingup", "jumping", "lying", "running", "sitting", "standing", "walking"]
LABEL_ENCODINGS = np.identity(len(ACTIVITIES))

# Index for raw data
RECORD_TIME = 0
ACTIVITY_ID = 1
ACC = 2
GYRO = 17
MAG = 32
LIG = 47
MODALITY_IDS = {"acc": ACC, "gyr": GYRO, "mag": MAG, "lig": LIG}
MODALITY_DIMS = {"acc": 3, "gyr": 3, "mag": 3, "lig": 1}

# Index for inertial sensor (only used for inertial sensors)
HEAD = 0
CHEST = 3
UPPERARM = 6
WAIST = 9
SHIN = 12
IMU_LOC_IDS = {"head": HEAD, "chest": CHEST, "upperarm": UPPERARM, "waist": WAIST, "shin": SHIN}
LIGHT_LOC_IDS = {"head": 0, "chest": 1, "upperarm": 2, "waist": 3, "shin": 4}

# Index for sensor dimensions
X = 0
Y = 1
Z = 2

# Mask for raw data extraction
ALL_PRESERVED_INDICES = [
    RECORD_TIME,
    ACC + HEAD + X,
    ACC + HEAD + Y,
    ACC + HEAD + Z,
    ACC + CHEST + X,
    ACC + CHEST + Y,
    ACC + CHEST + Z,
    ACC + UPPERARM + X,
    ACC + UPPERARM + Y,
    ACC + UPPERARM + Z,
    ACC + WAIST + X,
    ACC + WAIST + Y,
    ACC + WAIST + Z,
    ACC + SHIN + X,
    ACC + SHIN + Y,
    ACC + SHIN + Z,
    GYRO + HEAD + X,
    GYRO + HEAD + Y,
    GYRO + HEAD + Z,
    GYRO + CHEST + X,
    GYRO + CHEST + Y,
    GYRO + CHEST + Z,
    GYRO + UPPERARM + X,
    GYRO + UPPERARM + Y,
    GYRO + UPPERARM + Z,
    GYRO + WAIST + X,
    GYRO + WAIST + Y,
    GYRO + WAIST + Z,
    GYRO + SHIN + X,
    GYRO + SHIN + Y,
    GYRO + SHIN + Z,
    MAG + HEAD + X,
    MAG + HEAD + Y,
    MAG + HEAD + Z,
    MAG + CHEST + X,
    MAG + CHEST + Y,
    MAG + CHEST + Z,
    MAG + UPPERARM + X,
    MAG + UPPERARM + Y,
    MAG + UPPERARM + Z,
    MAG + WAIST + X,
    MAG + WAIST + Y,
    MAG + WAIST + Z,
    MAG + SHIN + X,
    MAG + SHIN + Y,
    MAG + SHIN + Z,
    LIG,
    LIG + 1,
    LIG + 2,
    LIG + 3,
    LIG + 4,
]


def extract_file_list(file_path):
    file_list = []
    filename_list = os.listdir(file_path)

    for filename in filename_list:
        if filename.endswith(".csv"):
            file_list.append(os.path.join(file_path, filename))

    return file_list


def extract_user_activity_id(file_name):
    """
    Extract user id from raw data file name.
    :param file_name:
    :return:
    """
    base = os.path.basename(file_name)
    user_id = os.path.splitext(base)[0]

    return user_id


def create_new_file(output_path, user_activity_id, user_activity_file_id):
    """
    Create a new file, decide its folder (train or test), and delete existing old file with same name.
    :param output_path:
    :param user_activity_id:
    :param user_activity_file_id:
    :param mode
    :return:
    """
    current_file_name = os.path.join(output_path, user_activity_id + "-" + str(user_activity_file_id) + ".pt")

    if os.path.exists(current_file_name):
        os.remove(current_file_name)

    return current_file_name


def split_array_with_overlap(input, num_interval, overlap_ratio):
    """Split the input array into num intervals with overlap ratio.

    Args:
        input (_type_): [800, 3]
        num_interval (_type_): 39
        overlap_ratio (_type_): 0.5
    """
    interval_len = int(len(input) // (1 + (num_interval - 1) * (1 - overlap_ratio)))

    splitted_input = []
    for start in range(0, len(input) - interval_len + 1, int((1 - overlap_ratio) * interval_len)):
        splitted_input.append(input[start : start + interval_len])
    splitted_input = np.array(splitted_input)

    return splitted_input


def interpolate_features(raw_data, start_time):
    """Padd and interpolate the given time-series features.

    Args:
        raw_data (_type_): _description_
        start_time (_type_): _description_
        sequence_len (_type_): _description_
    Return:
        Only sequences of sensor values, no time, no label
    """
    end_time = start_time + SEGMENT_SPAN - SEGMENT_SPAN / SEGMENT_SAMPLE_COUNT
    padded_times = np.concatenate([np.array([start_time]), raw_data[:, 0], np.array([end_time])])
    interp_times = np.linspace(start_time, end_time, SEGMENT_SAMPLE_COUNT)

    # interpolation
    interp_sensor_values = []
    for j in range(1, raw_data.shape[1]):
        padded_values = np.concatenate([np.array([raw_data[0, j]]), raw_data[:, j], np.array([raw_data[-1, j]])])
        interp_values = np.interp(interp_times, padded_times, padded_values)
        interp_sensor_values.append(interp_values)
    interp_sensor_values = np.stack(interp_sensor_values, axis=1)
    assert len(interp_sensor_values) == SEGMENT_SAMPLE_COUNT

    return interp_sensor_values


def extract_loc_mod_tensor(raw_data):
    """Extract the Tensor for a given location and sensor.
    We assume the data is interpolated before. No time dimension is included.

    Args:
        raw_data (_type_): _description_
        loc (_type_): _description_
        modality (_type_): _description_
    """
    assert len(raw_data) == SEGMENT_SAMPLE_COUNT
    num_dim = np.shape(raw_data)[1]

    # Step 1: Divide the segment into fixed-length intervals
    interval_sensor_values = split_array_with_overlap(raw_data, SEGMENT_INTERVAL_COUNT, OVERLAP_RATIO)

    # Step 2: Convert numpy array to tensor, and conver to [c. i, s] shape
    time_tensor = torch.from_numpy(interval_sensor_values).float()
    time_tensor = time_tensor.permute(2, 0, 1)

    # Step 3: Extract the FFT spectrum for each interval
    interval_spectrums = []
    for i in range(len(interval_sensor_values)):
        spectrums = []
        for j in range(num_dim):
            spectrum = np.fft.fft(interval_sensor_values[i, :, j])
            spectrums.extend([spectrum.real, spectrum.imag])

        interval_spectrum = np.stack(spectrums, axis=1)
        interval_spectrums.append(interval_spectrum)

    # combine all interval spectrums
    interval_spectrums = np.stack(interval_spectrums, axis=0)

    # Numpy array --> Torch tensor, in shape (i, s, c)
    freq_tensor = torch.from_numpy(interval_spectrums).float()

    # (i, s, c) --> (c, i, s) = (6 or 2, 9, 50)
    freq_tensor = torch.permute(freq_tensor, (2, 0, 1))

    return time_tensor, freq_tensor


def process_one_file(input_file, freq_output_path, time_output_path):
    """The function to process one input file.

    Args:
        input_file (_type_): _description_
        output_path (_type_): _description_
    """
    preserved_indices = ALL_PRESERVED_INDICES
    print("Processing: " + input_file)
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=" ")

        sample_start_time = 0
        sample_activity_id = -1
        segment = []
        user_activity_id = extract_user_activity_id(input_file)
        user_activity_file_id = 0

        freq_sample_file = create_new_file(freq_output_path, user_activity_id, user_activity_file_id)
        freq_sample = dict()
        freq_sample["flag"] = {loc: {} for loc in IMU_LOC_IDS}
        freq_sample["data"] = {loc: {} for loc in IMU_LOC_IDS}

        time_sample_file = create_new_file(time_output_path, user_activity_id, user_activity_file_id)
        time_sample = dict()
        time_sample["flag"] = {loc: {} for loc in IMU_LOC_IDS}
        time_sample["data"] = {loc: {} for loc in IMU_LOC_IDS}

        for record in csv_reader:
            if record:
                record_time = float(record[RECORD_TIME])
                record_activity_id = int(float(record[ACTIVITY_ID]))
                extracted_record = [float(record[i]) for i in preserved_indices]

                # Start a new segment when: enough time length has been reached, or different activity
                if sample_activity_id == -1:
                    sample_start_time = record_time
                    sample_activity_id = record_activity_id
                    freq_sample["label"] = torch.tensor(sample_activity_id).long()
                    time_sample["label"] = torch.tensor(sample_activity_id).long()
                else:
                    if (record_time - sample_start_time >= SEGMENT_SPAN - 1e-3) or (
                        not record_activity_id == sample_activity_id
                    ):
                        # save the segment if it is long enough
                        if len(segment) >= SEGMENT_SAMPLE_COUNT * 0.8:
                            # padding the data
                            segment = np.array(segment)
                            segment = interpolate_features(segment, sample_start_time)

                            # extract features for current segment
                            for loc in IMU_LOC_IDS:
                                for mod in MODALITY_IDS:
                                    if mod == "lig":
                                        start_col = MODALITY_IDS[mod] + LIGHT_LOC_IDS[loc] - 2
                                        end_col = start_col + MODALITY_DIMS[mod]
                                    else:
                                        start_col = MODALITY_IDS[mod] + IMU_LOC_IDS[loc] - 2
                                        end_col = start_col + MODALITY_DIMS[mod]

                                    # extract sensor feature
                                    loc_mod_data = segment[:, start_col:end_col]
                                    loc_mod_time_tensor, loc_mod_freq_tensor = extract_loc_mod_tensor(loc_mod_data)

                                    # save into sample
                                    freq_sample["data"][loc][mod] = loc_mod_freq_tensor
                                    freq_sample["flag"][loc][mod] = True

                                    time_sample["data"][loc][mod] = loc_mod_time_tensor
                                    time_sample["flag"][loc][mod] = True

                            # save the sample
                            torch.save(freq_sample, freq_sample_file)
                            torch.save(time_sample, time_sample_file)

                            # create a new file
                            user_activity_file_id += 1
                            freq_sample_file = create_new_file(
                                freq_output_path, user_activity_id, user_activity_file_id
                            )
                            time_sample_file = create_new_file(
                                time_output_path, user_activity_id, user_activity_file_id
                            )

                        # Start a new segment
                        segment = []
                        freq_sample = dict()
                        freq_sample["flag"] = {loc: {} for loc in IMU_LOC_IDS}
                        freq_sample["data"] = {loc: {} for loc in IMU_LOC_IDS}
                        sample_start_time = record_time
                        sample_activity_id = record_activity_id
                        freq_sample["label"] = torch.tensor(sample_activity_id).long()

                # Save current extracted record into current segment
                segment.append(extracted_record)


def process_one_file_wrapper(args):
    """The wrapper function for proces one file.

    Args:
        args (_type_): _description_
    """
    return process_one_file(*args)


def partition_and_extract_feature(input_path, freq_output_path, time_output_path):
    """
    Steps:
        1) Divide data from each user to 2 seconds segment, each as a data sample.
        2) FFT each time series segment and get frequency representation.
        3) Divide data into training and testing data, and saved into files.
    :param input_path:
    :param output_path:
    :return:
        Sample:
        - label: Tensor
        - flag
            - phone
                - audio: True
                - acc: False
        - data:
            -phone
                - audio: Tensor
                - acc: Tensor
    """
    input_file_list = extract_file_list(input_path)

    # Parallel pairing of samples
    pool = Pool(processes=cpu_count())
    args_list = [[input_file, freq_output_path, time_output_path] for input_file in input_file_list]
    pool.map(process_one_file_wrapper, args_list, chunksize=1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    input_path = "/home/sl29/data/RealWorld-HAR/processed-data/paired-data-5_pos-4_mod"
    freq_output_path = "/home/sl29/data/RealWorld-HAR/individual-freq-samples-5_pos-4_mod"
    time_output_path = "/home/sl29/data/RealWorld-HAR/individual-time-samples-5_pos-4_mod"

    for f in [freq_output_path, time_output_path]:
        if not os.path.exists(f):
            os.mkdir(f)

    start = time.time()

    partition_and_extract_feature(input_path, freq_output_path, time_output_path)

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
