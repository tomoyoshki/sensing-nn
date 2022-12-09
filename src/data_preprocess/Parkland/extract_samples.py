from genericpath import exists
import os
import time
import getpass
import torch
import pickle

import numpy as np
import torchaudio.functional as F
import torchaudio.transforms as T

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import cpu_count
from data_trunk import start_time_shift, end_time_shift
from wind_noise_generator import WindNoiseGenerator

"""
Configuration:
    1) Sampling frequency:
        - acoustic: 16K Hz
        - acc, seismic: 100 Hz
    2) Sample:
        - subject: subject ID
        - signal:
            -shake: {aud, ehz, acc}
    3) Audio file:
        - channel 0: mix
        - channel 1: single channel
    4) We simultaneously save the aligned time-series and the spectrogram for each channel.
"""

SEGMENT_SPAN = 2
INTERVAL_SPAN = 0.2
SEGMENT_OVERLAP_RATIO = 0.0
INTERVAL_OVERLAP_RATIO = 0.0
AUD_DOWNSAMPLE_RATE = 2

STD_THRESHOLD = 0

FREQS = {"audio": 16000 / AUD_DOWNSAMPLE_RATE, "seismic": 100, "acc": 100}
LABELS = {
    "Polaris": 0,
    "Warhog": 1,
    "Silverado": 2,
    "motor": 3,
    "tesla": 4,
    "mustang0528": 5,
    "walk": 6,
    "walk2": 6,
    "motor_noisy_run1": 3,
    "mustang_noisy_run1": 5,
    "walk_noisy_run1": 6,
    "motor_clean_run1": 3,
    "mustang_clean_run1": 5,
    "walk_clean_run1": 6,
}
SUBJECTS = {"rs3"}
PRESERVED_CLEAN_FOLDERS = {
    "motor",
    "mustang0528",
    "walk2",
    "tesla",
    "Polaris0150pm",
    "Polaris0215pm",
    "Polaris0235pm-NoLineOfSight",
    "Warhog1135am",
    "Warhog1149am",
    "Warhog-NoLineOfSight",
    "Silverado0255pm",
    "Silverado0315pm",
}

GC_FOLDERS = {
    "Polaris0150pm",
    "Polaris0215pm",
    "Polaris0235pm-NoLineOfSight",
    "Warhog1135am",
    "Warhog1149am",
    "Warhog-NoLineOfSight",
    "Silverado0255pm",
    "Silverado0315pm",
}


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


def folder_to_label(folder):
    """Convert a folder name to a label embedding.

    Args:
        folder (_type_): _description_
    """
    for label in LABELS:
        if label in folder:
            return label, LABELS[label]

    raise Exception(f"Invalid folder provided: {folder}")


def resample_numpy_array(raw_audio, orig_freq, new_freq):
    """Resample the given audio array with torchaudio APIs.

    Args:
        raw_audio (_type_): [time, channel]
    """
    # input, [time, channel] --> [channel, time]
    audio_tensor = torch.from_numpy(raw_audio)
    audio_tensor = torch.transpose(audio_tensor, 0, 1)

    # transform
    resampler = T.Resample(orig_freq, new_freq, dtype=float)
    new_audio_tensor = resampler(audio_tensor)
    # print(audio_tensor.shape, new_audio_tensor.shape)

    # output, [channel, time] --> [time, channel]
    new_audio_tensor = torch.transpose(new_audio_tensor, 0, 1)
    new_audio_array = new_audio_tensor.numpy()

    return new_audio_array


def extract_loc_mod_tensor(raw_data, segment_len, freq):
    """Extract the Tensor for a given location and sensor.
    We assume the data is interpolated before. No time dimension is included.

    Args:
        raw_data (_type_): _description_
        loc (_type_): _description_
        modality (_type_): _description_
    """
    assert len(raw_data) == segment_len * freq
    num_dim = np.shape(raw_data)[1]

    # Step 1: Divide the segment into fixed-length intervals, (i, s, c)
    interval_sensor_values = split_array_with_overlap(
        raw_data, INTERVAL_OVERLAP_RATIO, interval_len=int(INTERVAL_SPAN * freq)
    )

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


def process_one_sample(sample, label_id, folder, shake, freq_output_path, time_output_path, add_wind=False):
    """Process and save a sample.

    Args:
        sample (_type_): _description_
        folder (_type_): Contains labels and runs.
        shake (_type_): _description_
        output_path (_type_): _description_
    """
    id = sample["id"]

    if add_wind:
        wind_suffix = "_wind"
    else:
        wind_suffix = ""

    if shake is not None:
        freq_output_file = os.path.join(freq_output_path, f"{folder}_{shake}{wind_suffix}_{id}.pt")
        time_output_file = os.path.join(time_output_path, f"{folder}_{shake}{wind_suffix}_{id}.pt")
    else:
        freq_output_file = os.path.join(freq_output_path, f"{folder}{wind_suffix}_{id}.pt")
        time_output_file = os.path.join(time_output_path, f"{folder}{wind_suffix}_{id}.pt")

    time_sample = {
        "label": torch.tensor(label_id).long(),
        "flag": {},
        "data": {},
    }

    freq_sample = {
        "label": torch.tensor(label_id).long(),
        "flag": {},
        "data": {},
    }

    # extract modality tensor
    for loc in sample["signal"]:
        # freq placeholders
        freq_sample["data"][loc] = dict()
        freq_sample["flag"][loc] = dict()

        # time placeholders
        time_sample["data"][loc] = dict()
        time_sample["flag"][loc] = dict()

        for mod in sample["signal"][loc]:
            time_tensor, freq_tensor = extract_loc_mod_tensor(sample["signal"][loc][mod], SEGMENT_SPAN, FREQS[mod])

            # save freq sample
            freq_sample["data"][loc][mod] = freq_tensor
            freq_sample["flag"][loc][mod] = True

            # save tiem sample
            time_sample["data"][loc][mod] = time_tensor
            time_sample["flag"][loc][mod] = True

    # save the sample
    torch.save(freq_sample, freq_output_file)
    torch.save(time_sample, time_output_file)


def process_one_sample_wrapper(args):
    """Wrapper function for process a sample"""
    return process_one_sample(*args)


def process_one_shake(
    folder,
    shake,
    label_id,
    input_path,
    freq_output_path,
    time_output_path,
    add_wind=False,
    wind_generator=None,
):
    """Process a single folder.

    Args:
        input_folder (_type_): _description_
        output_path (_type_): _description_
    """
    # Step 1: Loading original files
    shake_path = os.path.join(input_path, folder, shake)

    start_second = start_time_shift[folder][shake]
    end_second = end_time_shift[folder][shake]
    print(f"Processing: {shake_path}")

    # load the audio, the file names are different for data collected from Parkland and GC
    if "aud16000.csv" in os.listdir(shake_path):
        audio_file = "aud16000.csv"
    else:
        audio_file = "aud.csv"
    raw_audio = np.loadtxt(os.path.join(shake_path, audio_file), dtype=float, delimiter=",")
    if raw_audio.ndim > 1:
        raw_audio = raw_audio[:, 0]
    raw_audio = np.expand_dims(raw_audio, axis=1)
    raw_audio = raw_audio[16000 * start_second : len(raw_audio) - 16000 * end_second]

    # add wind noise to the audio
    if add_wind:
        wind_noise = wind_generator.generate(len(raw_audio) / 16000, fs=16000, type="gusts")
        wind_noise = np.expand_dims(wind_noise, axis=1)
        print(raw_audio.shape, wind_noise.shape, wind_noise.min(), wind_noise.max())
        raw_audio += wind_noise * 80000
        print(raw_audio.shape, wind_noise.shape, wind_noise.min(), wind_noise.max())

    # resample the audio
    if AUD_DOWNSAMPLE_RATE > 1:
        raw_audio = resample_numpy_array(raw_audio, 16000, int(16000 / AUD_DOWNSAMPLE_RATE))

    # TODO: More preprocessing steps on the audio data

    # load the seismic data
    raw_seismic = np.loadtxt(os.path.join(shake_path, "ehz.csv"), dtype=float, delimiter=" ")
    if raw_seismic.ndim > 1:
        raw_seismic = raw_seismic[:, 0]
    raw_seismic = np.expand_dims(raw_seismic, axis=1)
    raw_seismic = raw_seismic[FREQS["seismic"] * start_second : len(raw_seismic) - FREQS["seismic"] * end_second]
    print(folder, shake, np.shape(raw_audio), np.shape(raw_seismic))

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
    print(f"Processing and saving individual samples: {folder, shake, len(sample_list)}")
    pool = Pool(max_workers=cpu_count())
    args_list = [[sample, label_id, folder, shake, freq_output_path, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.shutdown()


def process_one_shake_wrapper(args):
    """Wrapper function for procesing one folder."""
    return process_one_shake(*args)


if __name__ == "__main__":
    username = getpass.getuser()
    input_path = f"/home/{username}/data/Parkland/raw_data"
    add_wind = True

    if not add_wind:
        freq_output_path = f"/home/{username}/data/Parkland/individual_freq_samples"
        time_output_path = f"/home/{username}/data/Parkland/individual_time_samples"
    else:
        freq_output_path = f"/home/{username}/data/Parkland/individual_freq_samples_with_wind"
        time_output_path = f"/home/{username}/data/Parkland/individual_time_samples_with_wind"

    for f in [freq_output_path, time_output_path]:
        if not os.path.exists(f):
            os.mkdir(f)

    folders_to_process = []
    for e in os.listdir(input_path):
        if e in PRESERVED_CLEAN_FOLDERS:
            folders_to_process.append(e)

    # wind generator
    wind_generator = WindNoiseGenerator("wind_noise_model_v1.1.mat")

    # extract args list
    args_list = []
    for folder in folders_to_process:
        shake_list = os.listdir(os.path.join(input_path, folder))
        label, label_id = folder_to_label(folder)
        if folder in GC_FOLDERS:
            args_list.append(
                [
                    folder,
                    "rs1",
                    label_id,
                    input_path,
                    freq_output_path,
                    time_output_path,
                    add_wind,
                    wind_generator,
                ]
            )
        else:
            for shake in shake_list:
                # skip non folder items
                if not os.path.isdir(os.path.join(input_path, folder, shake)) or (shake not in SUBJECTS):
                    continue
                else:
                    args_list.append(
                        [
                            folder,
                            shake,
                            label_id,
                            input_path,
                            freq_output_path,
                            time_output_path,
                            add_wind,
                            wind_generator,
                        ]
                    )
            # break

    start = time.time()

    # for folder in folders_to_process:
    # process_one_folder(folder, input_path, output_path)

    pool = Pool(max_workers=cpu_count())
    pool.map(process_one_shake_wrapper, args_list, chunksize=1)
    pool.shutdown()
    # pool.close()
    # pool.join()

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
