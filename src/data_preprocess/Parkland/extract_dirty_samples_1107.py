import os
import time
import getpass

import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import cpu_count
from dirty_data_trunk import dirty_end_time_shift, dirty_start_time_shift
from extract_samples import process_one_sample_wrapper, resample_numpy_array, split_array_with_overlap, folder_to_label

SEGMENT_SPAN = 2
INTERVAL_SPAN = 0.2
SEGMENT_OVERLAP_RATIO = 0.0
INTERVAL_OVERLAP_RATIO = 0.0
AUD_DOWNSAMPLE_RATE = 2

STD_THRESHOLD = 0

FREQS = {"audio": 16000, "seismic": 100, "acc": 100}

PRESERVED_DIRTY_FOLDERS = {
    "motor_noisy_run1",
    "mustang_noisy_run1",
    "walk_noisy_run1",
}

PRESERVED_CLEAN_FOLDERS = {
    "motor_clean_run1",
    "mustang_clean_run1",
    "walk_clean_run1",
}


def process_one_folder(folder, label_id, input_path, freq_output_path, time_output_path):
    """Process a single folder.

    Args:
        input_folder (_type_): _description_
        output_path (_type_): _description_
    """
    # Step 1: Loading original files
    data_path = os.path.join(input_path, folder)
    print(f"Processing: {data_path}")

    # select the file
    for f in os.listdir(data_path):
        if "AUD" in f:
            audio_file = f
        elif "EHZ" in f:
            seismic_file = f

    # load the audio, the file names are different for data collected from Parkland and GC
    raw_audio = np.loadtxt(os.path.join(data_path, audio_file), dtype=float, delimiter=" ", skiprows=1)[:-1]
    if raw_audio.ndim > 1:
        raw_audio = raw_audio[:, 0]
    raw_audio = np.expand_dims(raw_audio, axis=1)

    # load the seismic data
    raw_seismic = np.loadtxt(os.path.join(data_path, seismic_file), dtype=str, delimiter=" ")[:-1]
    if raw_seismic.ndim > 1:
        raw_seismic = raw_seismic[:, 0].astype(float)
    raw_seismic = np.expand_dims(raw_seismic, axis=1)

    # align the audio and seismic data
    len_sec_audio = raw_audio.shape[0] // (SEGMENT_SPAN * FREQS["audio"]) * SEGMENT_SPAN
    len_sec_seismic = raw_seismic.shape[0] // (SEGMENT_SPAN * FREQS["seismic"]) * SEGMENT_SPAN
    len_sec = min(len_sec_audio, len_sec_seismic)
    print(len_sec_audio, len_sec_seismic, len_sec)
    raw_audio = raw_audio[: len_sec * FREQS["audio"]]
    raw_seismic = raw_seismic[: len_sec * FREQS["seismic"]]
    print(folder, np.shape(raw_audio), np.shape(raw_seismic))

    # resample the audio
    if AUD_DOWNSAMPLE_RATE > 1:
        raw_audio = resample_numpy_array(raw_audio, FREQS["audio"], int(FREQS["audio"] / AUD_DOWNSAMPLE_RATE))

    # Step 2: Partition into individual samples
    splitted_data = {"audio": [], "seismic": [], "acc": []}
    splitted_data["audio"] = split_array_with_overlap(
        raw_audio, SEGMENT_OVERLAP_RATIO, interval_len=SEGMENT_SPAN * FREQS["audio"] / AUD_DOWNSAMPLE_RATE
    )
    splitted_data["seismic"] = split_array_with_overlap(
        raw_seismic, SEGMENT_OVERLAP_RATIO, interval_len=SEGMENT_SPAN * FREQS["seismic"]
    )

    # prepare the individual samples
    sample_list = []
    for i in range(len(splitted_data["seismic"])):
        # Filter the data of parkpand according to their std
        if label_id not in [0, 1, 2]:
            if np.std(splitted_data["seismic"][i]) < STD_THRESHOLD:
                continue

        sample = {"id": i, "signal": {"shake": {}}}
        sample["signal"]["shake"]["audio"] = splitted_data["audio"][i]
        sample["signal"]["shake"]["seismic"] = splitted_data["seismic"][i]
        if len(splitted_data["seismic"][i]) < SEGMENT_SPAN * FREQS["seismic"]:
            continue
        else:
            sample_list.append(sample)

    # Step 3: Parallel processing and saving of the invidual samples
    print(f"Processing and saving individual samples: {folder,  len(sample_list)}")
    pool = Pool(max_workers=cpu_count())
    args_list = [[sample, label_id, folder, None, freq_output_path, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.shutdown()


def process_one_folder_wrapper(args):
    """Wrapper function for procesing one folder."""
    process_one_folder(*args)


if __name__ == "__main__":
    username = getpass.getuser()
    clean = False
    input_path = f"/home/{username}/data/Parkland_1107/raw_data"

    if clean:
        freq_output_path = f"/home/{username}/data/Parkland_1107/individual_clean_freq_samples"
        time_output_path = f"/home/{username}/data/Parkland_1107/individual_clean_time_samples"
        PRESERVED_FOLDERS = PRESERVED_CLEAN_FOLDERS
    else:
        freq_output_path = f"/home/{username}/data/Parkland_1107/individual_noisy_freq_samples"
        time_output_path = f"/home/{username}/data/Parkland_1107/individual_noisy_time_samples"
        PRESERVED_FOLDERS = PRESERVED_DIRTY_FOLDERS

    for f in [freq_output_path, time_output_path]:
        if not os.path.exists(f):
            os.mkdir(f)

    folders_to_process = []
    for e in os.listdir(input_path):
        if e in PRESERVED_FOLDERS:
            folders_to_process.append(e)

    # extract args list
    args_list = []
    for folder in folders_to_process:
        label, label_id = folder_to_label(folder)
        args_list.append([folder, label_id, input_path, freq_output_path, time_output_path])

    start = time.time()

    pool = Pool(max_workers=cpu_count())
    pool.map(process_one_folder_wrapper, args_list, chunksize=1)
    pool.shutdown()
    # pool.close()
    # pool.join()

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
