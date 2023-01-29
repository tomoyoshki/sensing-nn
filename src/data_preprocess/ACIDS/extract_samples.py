import os
import time
import torch
import numpy as np

from meta_loader import load_meta
from scipy.io import loadmat

from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import cpu_count

"""
Configuration:
    1) Sampling frequency:
        - acoustic: 1025.641 Hz
        - acc, seismic: 1025.641 Hz
    4) We save the aligned time-series for each sensor in shape [channel, interval, spectrum (time series)].
"""

SEGMENT_SPAN = 1
INTERVAL_SPAN = 0.25
SEGMENT_OVERLAP_RATIO = 0.0
INTERVAL_OVERLAP_RATIO = 0.5
STD_THRESHOLD = 0

FREQS = {"audio": 1024, "seismic": 1024}


def split_array_with_overlap(input, overlap_ratio, interval_len=None, num_interval=None):
    """Split the input array into num intervals with overlap ratio.

    Args:
        input (_type_): [freq * sec, channel]
        num_interval (_type_): 39
        overlap_ratio (_type_): 0.5
    """
    assert (interval_len is not None) or (num_interval is not None)

    if interval_len is None:
        interval_len = int(len(input) // (1 + (num_interval - 1) * (1 - overlap_ratio)))
    else:
        interval_len = int(interval_len)

    splitted_input = []
    for start in range(0, int(len(input) - interval_len + 1), int((1 - overlap_ratio) * interval_len)):
        interval = input[start : start + interval_len]

        # only prserve data wth complete length
        if len(interval) == interval_len:
            splitted_input.append(input[start : start + interval_len])
    splitted_input = np.array(splitted_input)

    return splitted_input


def extract_loc_mod_tensor(raw_data, segment_len, freq):
    """Extract the Tensor for a given location and sensor.
    We assume the data is interpolated before. No time dimension is included.

    Args:
        raw_data (_type_): _description_
        loc (_type_): _description_
        modality (_type_): _description_
    """
    assert len(raw_data) == segment_len * freq

    # Step 1: Divide the segment into fixed-length intervals, (i, s, c)
    interval_sensor_values = split_array_with_overlap(
        raw_data,
        INTERVAL_OVERLAP_RATIO,
        interval_len=int(INTERVAL_SPAN * freq),
    )

    # Step 2: Convert numpy array to tensor, and convert to [c. i, s] shape
    time_tensor = torch.from_numpy(interval_sensor_values).float()
    time_tensor = time_tensor.permute(2, 0, 1)

    return time_tensor


def process_one_sample(sample, labels, mat_file, time_output_path):
    """Process and save a sample.

    Args:
        sample (_type_): _description_
        folder (_type_): Contains labels and runs.
        shake (_type_): _description_
        output_path (_type_): _description_
    """
    id = sample["id"]
    time_output_file = os.path.join(time_output_path, f"{mat_file[:-4]}_{id}.pt")

    time_sample = {
        "label": {
            "vehicle_type": torch.tensor(labels["vehicle"]).float(),
            "terrain": torch.tensor(labels["terrain"]).float(),
            "speed": torch.tensor(labels["speed"]).float(),
            "distance": torch.tensor(labels["distance"]).float(),
        },
        "flag": {},
        "data": {},
    }

    # extract modality tensor
    for loc in sample["signal"]:
        # time placeholders
        time_sample["data"][loc] = dict()
        time_sample["flag"][loc] = dict()

        for mod in sample["signal"][loc]:
            time_tensor = extract_loc_mod_tensor(sample["signal"][loc][mod], SEGMENT_SPAN, FREQS[mod])

            # save tiem sample
            time_sample["data"][loc][mod] = time_tensor
            time_sample["flag"][loc][mod] = True

    # save the sample
    torch.save(time_sample, time_output_file)


def process_one_sample_wrapper(args):
    """Wrapper function for process a sample"""
    return process_one_sample(*args)


def process_one_mat(file, labels, input_path, time_output_path):
    """Process a single mat file.

    Args:
        labels: {
            "vehicle": vehicle_type,
            "terrain": terrain_type,
            "speed": float(speed),
            "distance": float(distance),
        }
    """
    # Step 1: Loading original files
    print(f"Processing: {file}")
    audio_file = os.path.join(input_path, "Acoustics", file)
    seismic_file = os.path.join(input_path, "Seismic", file[0:3] + "s" + file[3:])

    # load the audio, [channel, samples]
    raw_audio = loadmat(audio_file)["Output_data"]
    raw_audio = np.transpose(raw_audio, (1, 0))

    # load the seismic data
    raw_seismic = loadmat(seismic_file)["Output_data"]
    raw_seismic = np.transpose(raw_seismic, (1, 0))

    # Step 2: Partition into individual samples
    splitted_data = {"audio": [], "seismic": []}
    splitted_data["audio"] = split_array_with_overlap(
        raw_audio,
        SEGMENT_OVERLAP_RATIO,
        interval_len=SEGMENT_SPAN * FREQS["audio"],
    )
    splitted_data["seismic"] = split_array_with_overlap(
        raw_seismic,
        SEGMENT_OVERLAP_RATIO,
        interval_len=SEGMENT_SPAN * FREQS["seismic"],
    )

    # prepare the individual samples
    sample_list = []
    for i in range(len(splitted_data["seismic"])):
        sample = {"id": i, "signal": {"shake": {}}}
        sample["signal"]["shake"]["audio"] = splitted_data["audio"][i]
        sample["signal"]["shake"]["seismic"] = splitted_data["seismic"][i]

        if len(splitted_data["seismic"][i]) < SEGMENT_SPAN * FREQS["seismic"]:
            continue
        else:
            sample_list.append(sample)

    # Step 3: Parallel processing and saving of the invidual samples
    print(f"Processing and saving individual samples: {file, len(sample_list)}")
    pool = Pool(max_workers=cpu_count())
    args_list = [[sample, labels, file, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.shutdown()


def process_one_mat_wrapper(args):
    """Wrapper function for procesing one folder."""
    return process_one_mat(*args)


if __name__ == "__main__":
    input_path = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat"
    time_output_path = "/home/sl29/data/ACIDS/individual_time_samples_one_sec"
    meta_info = load_meta()

    if not os.path.exists(time_output_path):
        os.mkdir(time_output_path)

    # list the files to process
    files_to_process = []
    for e in os.listdir(os.path.join(input_path, "Acoustics")):
        if e.endswith(".mat"):
            files_to_process.append(e)

    # process the files in parallel
    args_list = []
    for file in files_to_process:
        if file in meta_info:
            args_list.append([file, meta_info[file], input_path, time_output_path])

    start = time.time()

    pool = Pool(max_workers=cpu_count())
    pool.map(process_one_mat_wrapper, args_list, chunksize=1)
    pool.shutdown()

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
