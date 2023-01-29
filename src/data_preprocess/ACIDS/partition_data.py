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


def partition_data(paired_data_path, output_path, train_files, val_files, test_files):
    """Partition the data according to the given ratio, using all-but-one strategy.
    We don't touch the processed data, but only save a new index file.

    Args:
        paired_data_path (_type_): _description_
        output_path (_type_): _description_
        train_ratio (_type_): _description_
    """
    # for users in training set, only preserve their data samples with complete modalities
    data_samples = os.listdir(paired_data_path)
    train_samples = []
    val_samples = []
    test_samples = []

    for sample in tqdm(data_samples):
        sample_file = os.path.basename(sample).split("_")[0]
        file_path = os.path.join(os.path.join(paired_data_path, sample))

        if sample_file in train_files:
            train_samples.append(file_path)
        else:
            val_samples.append(file_path)
            test_samples.append(file_path)

    # save the index file
    print(
        f"Number of training samples: {len(train_samples)}, \
        number of validation samples: {len(val_samples)}, \
        number of testing samples: {len(test_samples)}."
    )
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
    input_path = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat"
    paired_data_path = "/home/sl29/data/ACIDS/individual_time_samples_one_sec"
    output_path = f"/home/sl29/data/ACIDS/partition_index_one_sec2"
    meta_info = load_meta()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

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
    print(val_files)

    # partition the dta
    partition_data(paired_data_path, output_path, train_files, val_files, test_files)
