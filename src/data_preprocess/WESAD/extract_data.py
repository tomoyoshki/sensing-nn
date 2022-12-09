import os
import time
import getpass
from tkinter.messagebox import NO
import torch
import pickle

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

"""
Configuration:
    1) Sampling frequency:
        - Chest: All 700Hz
            - ECG
            - EDA
            - EMG
            - ACC (3-dim)
            - Respiration
        - Wrist:
            - ACC (3-dim): 32Hz
            - BVP: 64Hz
    2) Sample:
        - subject: subject ID
        - signal:
            -chest: {ACC, ECG, EDA, EMG, RESP}
            -wrist: {ACC, BVP}
            -label: in 700HZ, 
                * 0 = not defined / transient, 
                * 1 = baseline, 
                * 2 = stress, 
                * 3 = amusement, 
                * 4 = meditation, 
                * 5/6/7 = should be ignored in this dataset
    3) Data is synchronized and cleaned up, no missing value; no time information is included.
    4) ACC is not useful. We only consider 3 classes (1, 2, 3)
    5) We only use data from the chest.
"""

SEGMENT_SPAN = 3
INTERVAL_SPAN = 0.2
OVERLAP_RATIO = 0.0
LABEL_FREQ = 700

FREQS = {
    "chest": {"EMG": 700, "EDA": 700, "Resp": 700},
    # "wrist": {"ACC": 32, "BVP": 64},
}

PRESERVED_LABELS = {2: 0, 3: 1}


def extract_user_list(input_path):
    """Extract the user list in the given path.

    Args:
        input_path (_type_): _description_
    """
    user_list = []

    for e in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, e)):
            user_list.append(e)

    return user_list


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
    for start in range(0, len(input) - interval_len + 1, int((1 - overlap_ratio) * interval_len)):
        splitted_input.append(input[start : start + interval_len])
    splitted_input = np.array(splitted_input)

    return splitted_input


def extract_loc_mod_tensor(raw_data, sample_time_len, freq):
    """Extract the Tensor for a given location and sensor.
    We assume the data is interpolated before. No time dimension is included.

    Args:
        raw_data (_type_): _description_
        loc (_type_): _description_
        modality (_type_): _description_
    """
    assert len(raw_data) == sample_time_len * freq
    num_dim = np.shape(raw_data)[1]

    # Step 1: Divide the segment into fixed-length intervals
    interval_sensor_values = split_array_with_overlap(raw_data, OVERLAP_RATIO, interval_len=int(INTERVAL_SPAN * freq))

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


def process_one_sample(sample, user_id, freq_output_path, time_output_path):
    """Process and save a sample

    Args:
        sample (_type_): {signal: {loc: {mod: np.array}}, label: float, id: float}
        user_id (_type_): _description_
    """
    id = sample["id"]
    freq_output_file = os.path.join(freq_output_path, f"{user_id}_{id}.pt")
    time_output_file = os.path.join(time_output_path, f"{user_id}_{id}.pt")

    freq_sample = {
        "label": torch.tensor(PRESERVED_LABELS[sample["label"]]).long(),
        "flag": {},
        "data": {},
    }
    time_sample = {
        "label": torch.tensor(PRESERVED_LABELS[sample["label"]]).long(),
        "flag": {},
        "data": {},
    }

    for loc in sample["signal"]:
        # freq placeholders
        freq_sample["data"][loc] = dict()
        freq_sample["flag"][loc] = dict()

        # time placeholders
        time_sample["data"][loc] = dict()
        time_sample["flag"][loc] = dict()

        for mod in FREQS[loc]:
            if mod not in sample["signal"][loc]:
                print(loc, mod)
                freq_sample["flag"][loc][mod] = False
                time_sample["flag"][loc][mod] = False
            else:
                time_tensor, freq_tensor = extract_loc_mod_tensor(
                    sample["signal"][loc][mod], SEGMENT_SPAN, FREQS[loc][mod]
                )

                freq_sample["data"][loc][mod] = freq_tensor
                freq_sample["flag"][loc][mod] = True

                time_sample["data"][loc][mod] = time_tensor
                time_sample["flag"][loc][mod] = True

    # save the sample
    torch.save(freq_sample, freq_output_file)
    torch.save(time_sample, time_output_file)


def process_one_sample_wrapper(args):
    """Wrapper function for process a sample"""
    return process_one_sample(*args)


def process_one_user(input_path, freq_output_path, time_output_path, user_id):
    """The file to process one user.

    Args:
        input_path (_type_): _description_
        output_path (_type_): _description_
        user_id (_type_): _description_
    """
    user_input_file = os.path.join(input_path, user_id, f"{user_id}.pkl")
    with open(user_input_file, "rb") as f:
        all_samples = pickle.load(f, encoding="latin1")

    # split data into samples
    splitted_segments = {"label": [], "signal": {}}
    splitted_segments["label"] = split_array_with_overlap(
        all_samples["label"], OVERLAP_RATIO, interval_len=SEGMENT_SPAN * LABEL_FREQ
    )

    # split data nd extract features
    all_signals = all_samples["signal"]
    for loc in all_signals:
        if loc not in FREQS:
            continue
        else:
            splitted_segments["signal"][loc] = dict()
            for mod in all_signals[loc]:
                # skip some modalities
                if mod not in FREQS[loc]:
                    continue
                else:
                    splitted_segments["signal"][loc][mod] = split_array_with_overlap(
                        all_signals[loc][mod], OVERLAP_RATIO, interval_len=SEGMENT_SPAN * FREQS[loc][mod]
                    )

    # divide the data into list of individual samples
    sample_list = []
    sample_id = 0
    for i, label_array in enumerate(splitted_segments["label"]):
        unique, counts = np.unique(label_array, return_counts=True)
        # filter the classes
        if len(unique) > 1 or unique[0] not in PRESERVED_LABELS:
            continue
        else:
            sample = {"label": unique[0], "id": sample_id, "signal": {}}
            sample_id += 1
            for loc in splitted_segments["signal"]:
                sample["signal"][loc] = dict()
                for mod in splitted_segments["signal"][loc]:
                    sample["signal"][loc][mod] = splitted_segments["signal"][loc][mod][i]
            sample_list.append(sample)

    # parallel processing of the samples
    pool = Pool(processes=cpu_count())
    args_list = [[sample, user_id, freq_output_path, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.close()
    pool.join()


def process_one_user_wrapper(args):
    """The wrapper function for processing one user."""
    return process_one_user(*args)


if __name__ == "__main__":
    username = getpass.getuser()
    input_path = f"/home/{username}/data/WESAD/raw_data/WESAD"
    freq_output_path = f"/home/{username}/data/WESAD/freq_individual_samples"
    time_output_path = f"/home/{username}/data/WESAD/time_individual_samples"

    for f in [freq_output_path, time_output_path]:
        if not os.path.exists(f):
            os.mkdir(f)

    # extract user list
    user_list = extract_user_list(input_path)
    # print(user_list)

    # Parallel pairing of samples
    # pool = Pool(processes=cpu_count())
    # args_list = [[input_path, output_path, user_id] for user_id in user_list]
    # pool.map(process_one_user_wrapper, args_list, chunksize=1)
    # pool.close()
    # pool.join()

    # Serial processing of users
    start = time.time()

    for user_id in user_list:
        print(f"Processing user: {user_id}")
        process_one_user(input_path, freq_output_path, time_output_path, user_id)

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
