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

FREQS = {"audio": 16000 / AUD_DOWNSAMPLE_RATE, "seismic": 100, "acc": 100}
LABELS = {"Polaris": 0, "Warhog": 1, "Silverado": 2, "motor": 3, "tesla": 4, "mustang0528": 5, "walk2": 6}


PRESERVED_DIRTY_FOLDERS = {
    "motor",
    "mustang0528",
    "Polaris",
    "Silverado",
    "tesla",
    "walk2",
    "Warhog",
}


def process_one_folder(folder, label_id, input_path, freq_output_path, time_output_path):
    """Process a single folder.

    Args:
        input_folder (_type_): _description_
        output_path (_type_): _description_
    """
    # Step 1: Loading original files
    data_path = os.path.join(input_path, folder)
    start_second = dirty_start_time_shift[folder]
    end_second = dirty_end_time_shift[folder]
    print(f"Processing: {data_path}")

    # load the audio, the file names are different for data collected from Parkland and GC
    audio_file = "aud.csv"
    raw_audio = np.loadtxt(os.path.join(data_path, audio_file), dtype=float, delimiter=" ")
    if raw_audio.ndim > 1:
        raw_audio = raw_audio[:, 0]
    raw_audio = np.expand_dims(raw_audio, axis=1)
    raw_audio = raw_audio[16000 * start_second : len(raw_audio) - 16000 * end_second]

    # resample the audio
    if AUD_DOWNSAMPLE_RATE > 1:
        raw_audio = resample_numpy_array(raw_audio, 16000, int(16000 / AUD_DOWNSAMPLE_RATE))

    # load the seismic data
    raw_seismic = np.loadtxt(os.path.join(data_path, "ehz.csv"), dtype=float, delimiter=" ")
    if raw_seismic.ndim > 1:
        raw_seismic = raw_seismic[:, 0]
    raw_seismic = np.expand_dims(raw_seismic, axis=1)
    raw_seismic = raw_seismic[FREQS["seismic"] * start_second : len(raw_seismic) - FREQS["seismic"] * end_second]
    print(folder, np.shape(raw_audio), np.shape(raw_seismic))

    # Step 2: Partition into individual samples
    splitted_data = {"audio": [], "seismic": [], "acc": []}
    splitted_data["audio"] = split_array_with_overlap(
        raw_audio, SEGMENT_OVERLAP_RATIO, interval_len=SEGMENT_SPAN * FREQS["audio"]
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
    input_path = f"/home/{username}/data/Parkland_Dirty/raw_data"
    freq_output_path = f"/home/{username}/data/Parkland_Dirty/individual_freq_samples"
    time_output_path = f"/home/{username}/data/Parkland_Dirty/individual_time_samples"

    for f in [freq_output_path, time_output_path]:
        if not os.path.exists(f):
            os.mkdir(f)

    folders_to_process = []
    for e in os.listdir(input_path):
        if e in PRESERVED_DIRTY_FOLDERS:
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
