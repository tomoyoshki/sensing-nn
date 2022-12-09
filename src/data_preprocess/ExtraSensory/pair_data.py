import os
import time
import getpass
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import interpolate
from multiprocessing import Pool, cpu_count

# from joblib import Parallel, delayed

modality_rows = {
    "audio_naive": 430,
    "raw_acc": 800,  # 40hz
    "raw_magnet": 800,
}

VALID_RATIO = 0.95
NUM_INTERVALS = 20  # 1s window
OVERLAP_RATIO = 0.0


def get_raw_data_file(raw_data_path, user_id, timestamp, modality):
    """Get the raw sensor file for given (user_id, modality)

    The first column of raw sensor data is the relative time shift compared to the start.

    Args:
        user_id (_type_): _description_
        modality (_type_): _description_
    """
    modality_root_path = os.path.join(raw_data_path, modality)
    available_users = set(os.listdir(modality_root_path))
    if user_id not in available_users:
        return False, None

    modality_user_path = os.path.join(modality_root_path, user_id)
    available_timestamps = set(os.listdir(modality_user_path))

    if modality == "audio_naive":
        target_file_name = f"{timestamp}.sound.mfcc"
    elif modality == "raw_acc":
        target_file_name = f"{timestamp}.m_raw_acc.dat"
    elif modality == "raw_magnet":
        target_file_name = f"{timestamp}.m_raw_magnet.dat"
    elif modality == "watch_acc":
        target_file_name = f"{timestamp}.m_watch_acc.dat"
    else:
        raise Exception(f"Invalid modality provided: {modality}!")

    if target_file_name not in available_timestamps:
        return False, None
    else:
        return True, os.path.join(modality_user_path, target_file_name)


def get_dummy_data(modality):
    """Get the dummy data for the given modality.

    Args:
        modality (_type_): _description_
    Output:
        shape: [c, internval, spectrum]
    """
    if modality == "audio_naive":
        data = np.zeros([430, 13])
    else:
        data = np.zeros([NUM_INTERVALS, 2 * modality_rows[modality] // (NUM_INTERVALS + 1), 6])

    return data


def split_array_with_overlap(input, overlap_ratio, interval_len=None, num_interval=None):
    """Split the input array into num intervals with overlap ratio.

    Args:
        input (_type_): [700/32/64 * n (sec), 3/1]
        num_interval (_type_): 39
        overlap_ratio (_type_): 0.5
    """
    assert (interval_len is not None) or (num_interval is not None)

    if interval_len is None:
        interval_len = int(len(input) // (1 + (num_interval - 1) * (1 - overlap_ratio)))

    splitted_input = []
    for start in range(0, int(len(input) - interval_len + 1), int((1 - overlap_ratio) * interval_len)):
        splitted_input.append(input[start : start + interval_len])
    splitted_input = np.array(splitted_input)

    return splitted_input


def process_audio_feature(raw_data):
    """Preprocess the MFCC audio feature.
    If len(raw_data) < 0.5 * feature_len, mark missing;
    If 0.5 * feature_len <= len(raw_data) < feature_len, padding with 0;
    If len(raw_data) > feature_len, take the first 430 rows.

    Args:
        raw_data (_type_): _description_

    Return:
        Flag about whether the data is available, cleaned data
    """
    feature_len = modality_rows["audio_naive"]
    process_flag = True

    if len(raw_data) < VALID_RATIO * feature_len:
        process_flag = False
        processed_data = processed_data = get_dummy_data("audio_naive")
    elif len(raw_data) < feature_len:
        processed_data = get_dummy_data("audio_naive")
        processed_data[0 : len(raw_data)] = raw_data
    else:
        processed_data = raw_data[0:feature_len]

    # [430, 13] --> [390, 13]
    processed_data = processed_data[10:410, :]

    # data --> tensor, in shape (i, s)
    processed_tensor = torch.from_numpy(processed_data).float()

    # Regard 13 features as 13 dimensions: [390, 13] --> [39, 10, 13]
    processed_tensor = torch.reshape(processed_tensor, [NUM_INTERVALS, len(processed_data) // NUM_INTERVALS, -1])

    # add c dimension, (c, i, s) = (13, 39, 10)
    processed_tensor = torch.permute(processed_tensor, (2, 0, 1))

    return process_flag, processed_tensor


def process_inertial_feature(raw_data, modality, start_time, end_time):
    """Preprocess the inertial feature.
    1) If the length is too short, i.e., < 0.5 feature_len, mark missing;
    2) Otherwise, padding, or interpolate into the target length;
    3) Divide into intervals, and extract the frequency feature at each interval.

    Args:
        raw_data (_type_): The first column is time, column 2-4 --> x, y, z
        modality (_type_): _description_
    """
    feature_len = modality_rows[modality]
    process_flag = True

    # skip the preprocessing if the time span is too short, or lack time information
    if (len(raw_data) < VALID_RATIO * feature_len) or (raw_data.shape[1] != 4):
        process_flag = False
        processed_data = get_dummy_data(modality)
    else:
        # Step 1: Interpolation and extrapolation into target length
        padded_times = np.concatenate([np.array([start_time]), raw_data[:, 0], np.array([end_time])])
        interp_times = np.linspace(start_time, end_time, feature_len)
        interp_sensor_values = []
        for i in range(1, raw_data.shape[1]):
            padded_values = np.concatenate([np.array([raw_data[0, i]]), raw_data[:, i], np.array([raw_data[-1, i]])])
            interp_values = np.interp(interp_times, padded_times, padded_values)
            interp_sensor_values.append(interp_values)
        interp_sensor_values = np.stack(interp_sensor_values, axis=1)
        assert len(interp_sensor_values) == feature_len

        # Step 2: Divide the segment into fixed-length intervals
        interval_sensor_values = split_array_with_overlap(
            interp_sensor_values, OVERLAP_RATIO, num_interval=NUM_INTERVALS
        )

        # Step 3: Extract the FFT spectrum for each interval
        interval_spectrums = []
        for i in range(len(interval_sensor_values)):
            x_spectrum = np.fft.fft(interval_sensor_values[i, :, 0])
            y_spectrum = np.fft.fft(interval_sensor_values[i, :, 1])
            z_spectrum = np.fft.fft(interval_sensor_values[i, :, 2])

            interval_spectrum = np.stack(
                [x_spectrum.real, x_spectrum.imag, y_spectrum.real, y_spectrum.imag, z_spectrum.real, z_spectrum.imag],
                axis=1,
            )
            interval_spectrums.append(interval_spectrum)
        interval_spectrums = np.stack(interval_spectrums, axis=0)

        # Step 4: Save processed data
        processed_data = interval_spectrums

    # Numpy array --> Torch tensor, in shape (i, s, c)
    processed_tensor = torch.from_numpy(processed_data).float()

    # (i, s, c) --> (c, i, s) = (6, 39, 40)
    processed_tensor = torch.permute(processed_tensor, (2, 0, 1))

    return process_flag, processed_tensor


def pair_a_sample(raw_data_path, output_path, user_id, timestamp, modalities, labels):
    """Pair all modalities for the given (user_id, timestamp)

        - Phone audio: (~430) x 13, no time dimension contained, an extra "," at end of line
        - Watch acc: 25Hz * 20s * 3dim
        - Phone acc or mag: 40Hz * 20s * 3dim

    Args:
        raw_data_path (_type_): _description_
        user_id (_type_): _description_
        timestamp (_type_): _description_
    Return:
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
    sample = dict()
    sample["label"] = torch.tensor(np.argmax(labels)).long()
    sample["flag"] = {"phone": {}}
    sample["data"] = {"phone": {}}

    # read the modality and save them into the cache
    modality_to_raw_data = dict()
    min_time = np.inf
    max_time = -np.inf
    for modality in modalities:
        # identify the raw_data_file
        raw_flag, raw_data_file = get_raw_data_file(raw_data_path, user_id, timestamp, modality)

        # mark missing if file not exist
        if not raw_flag:
            sample["flag"]["phone"][modality] = False
            sample["data"]["phone"][modality] = torch.from_numpy(get_dummy_data(modality))
            continue

        if "audio" in modality:
            """Preprocess the audio data without extracing start and end time"""
            try:
                modality_raw_data = np.loadtxt(raw_data_file, delimiter=",", dtype=str)[:, :-1].astype(float)
            except:
                # Typo" 30067)3 in file: /home/sl29/data/ExtraSensory/raw_sensor_data/audio_naive/99B204C0-DD5C-4BB7-83E8-A37281B8D769/1444502547.sound.mfcc
                raise Exception(raw_data_file)
            process_flag, processed_tensor = process_audio_feature(modality_raw_data)
            sample["flag"]["phone"][modality] = process_flag
            sample["data"]["phone"][modality] = processed_tensor
        else:
            try:
                modality_raw_data = np.loadtxt(raw_data_file, delimiter=" ", dtype=float)
            except:
                raise Exception(raw_data_file)
            modality_to_raw_data[modality] = modality_raw_data
            modality_times = modality_raw_data[:, 0]
            min_modality_time, max_modality_time = np.min(modality_times), np.max(modality_times)
            min_time, max_time = min(min_time, min_modality_time), max(max_time, max_modality_time)
            # print(modality_raw_data.shape, min_time, max_time)

    # preprocess the acc/mag/watch-acc raw data and save the flag
    for modality in modalities:
        if (modality in sample["flag"]["phone"] and not sample["flag"]["phone"][modality]) or ("audio" in modality):
            continue
        else:
            process_flag, processed_tensor = process_inertial_feature(
                modality_to_raw_data[modality], modality, min_time, max_time
            )
            sample["flag"]["phone"][modality] = process_flag
            sample["data"]["phone"][modality] = processed_tensor

    # check the flags
    sample_flag = True
    for loc in sample["flag"]:
        for mod in sample["flag"][loc]:
            if not sample["flag"][loc][mod]:
                sample_flag = False

    # save the sample
    # print(sample["flag"]["phone"])
    if sample_flag:
        output_file = os.path.join(output_path, f"{user_id}_{timestamp}.pt")
        torch.save(sample, output_file)

    # test saved file
    # load_sample = torch.load(output_file)
    # print(load_sample["data"]["phone"])


def pair_a_sample_wrapper(args):
    """Wrapper function for pairing a sample."""
    return pair_a_sample(*args)


def process_a_user(raw_data_path, output_path, label_path, modalities, user_id):
    """Process the samples for a given user.

    Each (user_id, timestamp) is stored in a separate file, including both labels and multi-modal data.
    Args:
        raw_data_path (_type_): _description_
        label_path (_type_): _description_
        modalities (_type_): _description_
        user_id (_type_): _description_
    """
    print(f"-------------- Processing user: {user_id} --------------")

    # load the labels first
    user_label_file = os.path.join(label_path, f"{user_id}.csv")
    label_df = pd.read_csv(filepath_or_buffer=user_label_file, delimiter=",", header=0)
    label_array = label_df.to_numpy(dtype=float)
    timestamp_to_labels = dict(zip(label_array[:, 0].astype(int), label_array[:, 1:]))

    # Parallel pairing of samples
    pool = Pool(processes=cpu_count())
    args_list = [
        [raw_data_path, output_path, user_id, timestamp, modalities, timestamp_to_labels[timestamp]]
        for timestamp in timestamp_to_labels
    ]
    # Parallel(n_jobs=cpu_count() * 2)(delayed(pair_a_sample_wrapper)(e) for e in args_list)
    pool.map(pair_a_sample_wrapper, args_list, chunksize=1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    username = getpass.getuser()
    input_label_path = f"/home/{username}/data/ExtraSensory/clean_labels"
    raw_data_path = f"/home/{username}/data/ExtraSensory/raw_sensor_data"
    output_path = f"/home/{username}/data/ExtraSensory/paired_data"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # extract the modalities
    modalities = os.listdir(raw_data_path)

    # user list
    users = [e.split(".")[0] for e in os.listdir(input_label_path)]

    start = time.time()
    for user_id in tqdm(users):
        # if user_id != "9759096F-1119-4E19-A0AD-6F16989C7E1C":
        #     continue

        process_a_user(raw_data_path, output_path, input_label_path, modalities, user_id)
    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
