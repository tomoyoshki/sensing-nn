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
    1) Sensors: BVP 64 Hz, ACC 32 Hz
    2) Length: 8 seconds with 2 seconds shift
"""

SEGMENT_SPAN = 8
INTERVAL_SPAN = 1
INTERVAL_OVERLAP_RATIO = 0.0
SEGMENT_OVERLAP_RATIO = 0.75
LABEL_FREQ = 0.5


FREQS = {
    "wrist": {"ACC": 32, "BVP": 64},
}


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

    # Step 1: Divide the segment into fixed-length intervals
    interval_sensor_values = split_array_with_overlap(
        raw_data, INTERVAL_OVERLAP_RATIO, interval_len=int(INTERVAL_SPAN * freq)
    )

    # Step 2: Convert numpy array to tensor, and convert to [c. i, s] shape
    time_tensor = torch.from_numpy(interval_sensor_values).float()
    time_tensor = time_tensor.permute(2, 0, 1)

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
        "label": torch.tensor(sample["label"]).float(),
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
                time_tensor = extract_loc_mod_tensor(sample["signal"][loc][mod], SEGMENT_SPAN, FREQS[loc][mod])

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
    user_input_file = os.path.join(input_path, user_id, f"{user_id}.pkl")

    # read the pickle file
    with open(user_input_file, "rb") as f:
        user_datta = pickle.load(f, encoding="latin1")

    splitted_segments = {"label": [], "signal": {}}
    splitted_segments["label"] = user_datta["label"]
    splitted_segments["signal"]["wrist"] = {}

    # split data
    for mod in {"ACC", "BVP"}:
        splitted_segments["signal"]["wrist"][mod] = split_array_with_overlap(
            user_datta["signal"]["wrist"][mod],
            SEGMENT_OVERLAP_RATIO,
            interval_len=int(SEGMENT_SPAN * FREQS["wrist"][mod]),
        )

    # print(splitted_segments["label"].shape)
    # print(splitted_segments["signal"]["wrist"]["ACC"].shape)
    # print(splitted_segments["signal"]["wrist"]["BVP"].shape)

    # divide the data into list of individual samples
    sample_list = []
    sample_id = 0
    for i, label in enumerate(splitted_segments["label"]):
        # optional skipping
        if i % 2 == 0:
            continue

        # filter the classes
        sample = {"label": label, "id": sample_id, "signal": {}}
        sample_id += 1
        for loc in splitted_segments["signal"]:
            sample["signal"][loc] = dict()
            for mod in splitted_segments["signal"][loc]:
                sample["signal"][loc][mod] = splitted_segments["signal"][loc][mod][i]
        sample_list.append(sample)

    # parallel processing of the samples
    pool = Pool(processes=cpu_count())
    args_list = [[sample, user_id, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.close()
    pool.join()


def process_one_user_wrapper(args):
    """The wrapper function for processing one user."""
    return process_one_user(*args)


if __name__ == "__main__":
    username = getpass.getuser()
    input_path = f"/home/{username}/data/PPG_DaLiA/raw_data/PPG_FieldStudy"
    time_output_path = f"/home/{username}/data/PPG_DaLiA/time_individual_samples"

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
