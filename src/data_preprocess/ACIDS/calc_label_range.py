import os
import time
import json
import torch
import numpy as np

from meta_loader import load_meta
from scipy.io import loadmat

from extract_samples import split_array_with_overlap, SEGMENT_OVERLAP_RATIO, FREQS, SEGMENT_SPAN
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import cpu_count


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
    audio_energies = []
    for i in range(len(splitted_data["seismic"])):
        energy = np.sum(np.square(splitted_data["audio"][i].astype(np.int64)), axis=0)
        audio_energies.append(energy)

    # save the energy file
    np.savetxt(os.path.join(time_output_path, f"{file.split('.')[0]}_energy.txt"), audio_energies)


def process_one_mat_wrapper(args):
    """Wrapper function for procesing one folder."""
    return process_one_mat(*args)


def calc_label_ranges(energy_path, json_output, upper_db_threshold, lower_db_threshold):
    """Calculate the label ranges for each mat file."""
    # calculate the cutoff energy range
    min_max_energy = np.inf
    for file in os.listdir(energy_path):
        input_file = os.path.join(energy_path, file)
        audio_energies = np.loadtxt(input_file, dtype=np.float64)
        max_energy = np.max(audio_energies)
        if max_energy < min_max_energy:
            min_max_energy = max_energy

    signal_energy_treshold = min_max_energy * 10 ** (upper_db_threshold / 10)
    background_energy_threshold = min_max_energy * 10 ** (lower_db_threshold / 10)
    print("Cutoff energy: ", signal_energy_treshold, background_energy_threshold)

    # calculate the label ranges
    label_ranges = {}
    for file in os.listdir(energy_path):
        input_file = os.path.join(energy_path, file)
        mat_file = file.split("_")[0] + ".mat"
        audio_energies = np.loadtxt(input_file, dtype=np.float64)
        audio_energies = np.max(audio_energies, axis=1)
        background_flags = audio_energies < background_energy_threshold
        signal_flags = audio_energies > signal_energy_treshold
        mat_label_range = {"background": [], "cpa": []}
        for i, (signal_flag, background_flag) in enumerate(zip(signal_flags, background_flags)):
            start_id = int(i * FREQS["audio"] * SEGMENT_SPAN * SEGMENT_OVERLAP_RATIO)
            id_span = int(FREQS["audio"] * SEGMENT_SPAN)
            if background_flag:
                mat_label_range["background"].append([start_id, start_id + id_span])
            elif signal_flag:
                mat_label_range["cpa"].append([start_id, start_id + id_span])

        label_ranges[mat_file] = mat_label_range
        print(
            mat_file,
            ", valid samples: ",
            np.sum(1 - background_flags),
            ", background samples: ",
            np.sum(background_flags),
        )

    # save the label ranges
    with open(json_output, "w") as f:
        f.write(json.dumps(label_ranges, indent=4))


if __name__ == "__main__":
    input_path = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat"
    energy_path = "/home/sl29/data/ACIDS/mat_segment_energies"
    label_range_file = "/home/sl29/data/ACIDS/global_mat_label_range.json"
    meta_info = load_meta()
    upper_db_threshold = -4
    lower_db_threshold = -10

    # list the files to process
    args_list = []
    for e in os.listdir(os.path.join(input_path, "Acoustics")):
        if e.endswith(".mat") and e in meta_info:
            args_list.append([e, meta_info[e], input_path, energy_path])
    print(f"Valid mat file count: {len(args_list)}")

    start = time.time()
    pool = Pool(max_workers=cpu_count())
    pool.map(process_one_mat_wrapper, args_list, chunksize=1)
    pool.shutdown()

    # generate the label ranges
    calc_label_ranges(energy_path, label_range_file, upper_db_threshold, lower_db_threshold)
