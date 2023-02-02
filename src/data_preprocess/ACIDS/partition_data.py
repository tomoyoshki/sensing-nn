from cgi import test
import os
import torch
import random
import getpass

import numpy as np
from meta_loader import load_meta

from tqdm import tqdm


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


def truncate_data_samples(data_samples):
    """Remove the first few seconds and the last few seconds in the data"""
    truncated_data_samples = []
    start_length = 20
    end_length = 20

    # first iteration
    mat_to_max_ids = {}
    for sample in data_samples:
        sample_mat = os.path.basename(sample).split("_")[0]
        sample_id = int(os.path.basename(sample).split("_")[1].split(".")[0])

        if sample_mat not in mat_to_max_ids or sample_id > mat_to_max_ids[sample_mat]:
            mat_to_max_ids[sample_mat] = sample_id

    # second iteration
    for sample in data_samples:
        sample_mat = os.path.basename(sample).split("_")[0]
        sample_id = int(os.path.basename(sample).split("_")[1].split(".")[0])
        if start_length <= sample_id <= mat_to_max_ids[sample_mat] - end_length:
            truncated_data_samples.append(sample)

    return truncated_data_samples


def partition_data(option, paired_data_path, output_path, train_files, val_files, test_files):
    """Partition the data according to the given ratio, using all-but-one strategy.
    We don't touch the processed data, but only save a new index file.

    Args:
        paired_data_path (_type_): _description_
        output_path (_type_): _description_
        train_ratio (_type_): _description_
    """
    # for users in training set, only preserve their data samples with complete modalities
    data_samples = os.listdir(paired_data_path)

    # sample truncation
    data_samples = truncate_data_samples(data_samples)

    # split
    pretrain_samples = []
    train_samples = []
    val_samples = []
    test_samples = []

    for sample in tqdm(data_samples):
        sample_file = os.path.basename(sample).split("_")[0]
        file_path = os.path.join(os.path.join(paired_data_path, sample))

        load_sample = torch.load(file_path)
        if option != "vehicle_classification" and load_sample["label"]["vehicle_type"].item() == 0:
            continue

        if sample_file in train_files:
            pretrain_samples.append(file_path)
            if load_sample["label"]["vehicle_type"].item() == 0 and random.random() < 0.9:
                continue
            else:
                train_samples.append(file_path)
        else:
            if load_sample["label"]["vehicle_type"].item() == 0 and random.random() < 0.9:
                continue
            else:
                val_samples.append(file_path)
                test_samples.append(file_path)

    # save the index file
    print(
        f"Number of pretraining samples: {len(pretrain_samples)}, \
        number of training samples: {len(val_samples)}, \
        number of validation samples: {len(val_samples)}, \
        number of testing samples: {len(test_samples)}."
    )
    with open(os.path.join(output_path, "pretrain_index.txt"), "w") as f:
        for sample_file in pretrain_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "train_index.txt"), "w") as f:
        for sample_file in train_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "val_index.txt"), "w") as f:
        for sample_file in val_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "test_index.txt"), "w") as f:
        for sample_file in test_samples:
            f.write(sample_file + "\n")

    # Synchronize the data from INCAS --> Eugene
    cmd = f"rsync -av {output_path}/ eugene:{output_path}/"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    option = "speed_classification"
    input_path = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat"
    paired_data_path = "/home/sl29/data/ACIDS/individual_time_samples_one_sec"
    meta_info = load_meta()

    files_to_process = []
    for e in os.listdir(os.path.join(input_path, "Acoustics")):
        if e.endswith(".mat") and e in meta_info:
            files_to_process.append(e[:-4])

    train_files, val_files, test_files = [], [], []
    cover_flags = [False for _ in range(9)]
    random.shuffle(files_to_process)
    for e in files_to_process:
        class_id = int(e[2]) - 1
        if not cover_flags[class_id]:
            val_files.append(e)
            test_files.append(e)
            cover_flags[class_id] = True
        else:
            if random.random() < 0.95:
                train_files.append(e)
            else:
                val_files.append(e)
                test_files.append(e)

    for option in [
        "vehicle_classification",
        "speed_classification",
        "distance_classification",
        "terrain_classification",
    ]:
        print(f"Processing option: {option}.")
        output_path = f"/home/sl29/data/ACIDS/partition_index_{option}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # partition the dta
        partition_data(option, paired_data_path, output_path, train_files, val_files, test_files)
