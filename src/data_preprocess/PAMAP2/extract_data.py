import os
import time
import getpass
import pickle
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

"""
Configuration:
    1) Sensors: Acc, Gyro, Mag; 3 locations: hand, chest, ankle
    2) Length: 2s per sample
    3) The IMU sensory data contains the following columns:
        1. temperature
        2-4. 3D-acceleration data (ms-2), scale: Â±16g, resolution: 13-bit
        5-7. 3D-acceleration data (ms-2), scale: Â±6g, resolution: 13-bit
        8-10. 3D-gyroscope data (rad/s)
        11-13. 3D-magnetometer data (Î¼T)
        14-17. orientation (invalid in this data collection)
"""

SEGMENT_SPAN = 2
INTERVAL_SPAN = 0.4
INTERVAL_OVERLAP_RATIO = 0.5
SEGMENT_OVERLAP_RATIO = 0.0
FREQ = 100

LOCS = {"hand"}
MODS = {"Acc", "Gyro", "Mag"}
LOC_IDS = {"hand": 3, "chest": 20, "android": 37}
MOD_IDS = {"Acc": 1, "Gyro": 7, "Mag": 10}
MOD_DIM = {"Acc": 3, "Gyro": 3, "Mag": 3}

FREQS = {
    "hand": {"Acc": 100, "Gyro": 100, "Mag": 100},
}

PRESERVED_LABELS = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    9: 7,
    10: 8,
    11: 9,
    12: 10,
    13: 11,
    16: 12,
    17: 13,
    18: 14,
    19: 15,
    20: 16,
    24: 17,
}


def extract_user_list(input_path):
    """Extract the user list in the given path."""
    user_list = []
    for e in os.listdir(input_path):
        user_list.append(e.split(".")[0])

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

    # Step 1: Divide the segment into fixed-length intervals
    interval_sensor_values = split_array_with_overlap(
        raw_data, INTERVAL_OVERLAP_RATIO, interval_len=int(INTERVAL_SPAN * freq)
    )

    # Step 2: Convert numpy array to tensor, and convert to [c. i, s] shape
    time_tensor = torch.from_numpy(interval_sensor_values).float()
    time_tensor = time_tensor.permute(2, 0, 1)

    # Step 3: Replace NaN with 0
    time_tensor = torch.nan_to_num(time_tensor, nan=0.0)

    return time_tensor


def process_one_sample(sample, user_id, time_output_path):
    """Process and save a sample

    Args:
        sample (_type_): {signal: {loc: {mod: np.array}}, label: float, id: float}
        user_id (_type_): _description_
    """
    id = sample["id"]
    time_output_file = os.path.join(time_output_path, f"{user_id}_{id}.pt")

    time_sample = {
        "label": torch.tensor(sample["label"]).long(),
        "flag": {},
        "data": {},
    }

    for loc in sample["signal"]:
        # time placeholders
        time_sample["data"][loc] = dict()
        time_sample["flag"][loc] = dict()

        for mod in FREQS[loc]:
            if mod not in sample["signal"][loc]:
                print(loc, mod)
                time_sample["flag"][loc][mod] = False
            else:
                time_tensor = extract_loc_mod_tensor(sample["signal"][loc][mod], SEGMENT_SPAN, FREQ)

                time_sample["data"][loc][mod] = time_tensor
                time_sample["flag"][loc][mod] = True

    # save the sample
    torch.save(time_sample, time_output_file)


def process_one_sample_wrapper(args):
    """Wrapper function for process a sample"""
    return process_one_sample(*args)


def process_one_user(input_path, time_output_path, user_id):
    """
    The function to process one user.
    """
    user_input_file = os.path.join(input_path, f"{user_id}.dat")

    # read the pickle file
    user_data = np.genfromtxt(user_input_file, delimiter=" ")

    # split data
    splitted_uesr_data = split_array_with_overlap(
        user_data,
        SEGMENT_OVERLAP_RATIO,
        interval_len=int(SEGMENT_SPAN * FREQ),
    )
    print(splitted_uesr_data.shape)

    # divide the data into list of individual samples
    sample_list = []
    sample_id = 0
    for i, sample_data in enumerate(splitted_uesr_data):
        # remove cross-boundary samples
        unique, counts = np.unique(sample_data[:, 1], return_counts=True)
        if len(unique) > 1 or unique[0] not in PRESERVED_LABELS:
            continue
        else:
            label = PRESERVED_LABELS[unique[0]]

        # filter the classes
        sample = {"label": label, "id": sample_id, "signal": {}}
        sample_id += 1
        for loc in LOCS:
            sample["signal"][loc] = dict()
            for mod in MODS:
                sample["signal"][loc][mod] = sample_data[
                    :, LOC_IDS[loc] + MOD_IDS[mod] : LOC_IDS[loc] + MOD_IDS[mod] + MOD_DIM[mod]
                ]
        sample_list.append(sample)

    # parallel processing of the samples
    pool = Pool(processes=cpu_count())
    args_list = [[sample, user_id, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    username = getpass.getuser()
    input_path = f"/home/{username}/data/PAMAP2/raw_data/PAMAP2_Dataset/Protocol"
    time_output_path = f"/home/{username}/data/PAMAP2/time_individual_samples"

    if not os.path.exists(time_output_path):
        os.makedirs(time_output_path)

    # extract user list
    user_list = extract_user_list(input_path)

    # Serial processing of users
    start = time.time()
    for user in user_list:
        process_one_user(input_path, time_output_path, user)
    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
