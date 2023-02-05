import os
import json
import torch
import random
import getpass
import collections

import numpy as np
from meta_loader import load_meta
from tqdm import tqdm
from ACIDS_file_partitions import *


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


def random_parition_files(files_to_process):
    """Randomly partition the files into train, val, and test sets."""
    train_files, val_files, test_files = [], [], []
    train_cover_flags = [False for _ in range(9)]
    val_cover_flags = [False for _ in range(9)]
    # test_cover_flags = [False for _ in range(9)]

    random.shuffle(files_to_process)
    for e in files_to_process:
        class_id = int(e[2]) - 1
        if not val_cover_flags[class_id]:
            val_files.append(e)
            val_cover_flags[class_id] = True
        # elif not test_cover_flags[class_id]:
        #     test_files.append(e)
        #     test_cover_flags[class_id] = True
        elif not train_cover_flags[class_id]:
            train_files.append(e)
            train_cover_flags[class_id] = True
        else:
            rand_num = random.random()
            if rand_num < 0.9:
                train_files.append(e)
            # elif 0.8 <= rand_num < 0.9:
            #     val_files.append(e)
            else:
                val_files.append(e)

    test_files = val_files

    return train_files, val_files, test_files


def partition_data(task, paired_data_path, output_path, train_files, val_files, test_files):
    """Partition the data according to the given ratio, using all-but-one strategy.
    We don't touch the processed data, but only save a new index file.

    Args:
        paired_data_path (_type_): _description_
        output_path (_type_): _description_
        train_ratio (_type_): _description_
    """
    # for users in training set, only preserve their data samples with complete modalities
    data_samples = os.listdir(paired_data_path)
    bg_ratio = 1

    # sample truncation
    # data_samples = truncate_data_samples(data_samples)

    # split
    pretrain_samples = []
    train_samples = []
    val_samples = []
    test_samples = []
    train_class_count = {}
    val_class_count = {}
    test_class_count = {}

    for sample in tqdm(data_samples):
        sample_file = os.path.basename(sample).split("_")[0]
        file_path = os.path.join(os.path.join(paired_data_path, sample))
        load_sample = torch.load(file_path)
        vehicle_type = load_sample["label"]["vehicle_type"].item()

        # background class is not used in speed/terrain/distance classification
        if task == "vehicle_classification":
            if sample_file in train_files:
                pretrain_samples.append(file_path)
                if vehicle_type > 0 or random.random() < bg_ratio:
                    if vehicle_type not in train_class_count:
                        train_class_count[vehicle_type] = 1
                    else:
                        train_class_count[vehicle_type] += 1
                    train_samples.append(file_path)
            elif sample_file in val_files:
                if vehicle_type > 0 or random.random() < bg_ratio:
                    if vehicle_type not in val_class_count:
                        val_class_count[vehicle_type] = 1
                    else:
                        val_class_count[vehicle_type] += 1
                    val_samples.append(file_path)
                    # else:
                    #     if vehicle_type > 0 or random.random() < bg_ratio:
                    #         if vehicle_type not in test_class_count:
                    #             test_class_count[vehicle_type] = 1
                    #         else:
                    #             test_class_count[vehicle_type] += 1
                    test_samples.append(file_path)

        else:
            if vehicle_type == 0:
                continue

            # extract class id
            if task == "speed_classification":
                class_id = load_sample["label"]["speed"].item()
            elif task == "distance_classification":
                class_id = load_sample["label"]["distance"].item()
            else:
                class_id = load_sample["label"]["terrain"].item()

            if class_id < 0:
                continue

            if sample_file in train_files:
                train_samples.append(file_path)
            elif sample_file in val_files:
                val_samples.append(file_path)
                # else:
                test_samples.append(file_path)

    # Print class distribution for training set of vehicle classification
    if task == "vehicle_classification":
        print(f"Class distribution for training set: {dict(sorted(train_class_count.items()))}")
        print(f"Class distribution for validation set: {dict(sorted(val_class_count.items()))}")
        print(f"Class distribution for testing set: {dict(sorted(test_class_count.items()))}")

    # save the index file
    print(
        f"Pretraining samples: {len(pretrain_samples)}, \
        training samples: {len(train_samples)}, \
        validation samples: {len(val_samples)}, \
        testing samples: {len(test_samples)}."
    )
    if len(pretrain_samples) > 0:
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

    # save the class sample distribution
    np.savetxt(
        os.path.join(output_path, "train_class_count.txt"),
        np.array(list(dict(sorted(train_class_count.items())).values())).astype(int),
    )

    # Synchronize the data from INCAS --> Eugene
    cmd = f"rsync -av {output_path}/ eugene:{output_path}/"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parition_option = "random"
    input_path = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat"
    paired_data_path = "/home/sl29/data/ACIDS/individual_time_samples_one_sec"
    meta_info = load_meta()

    # load the label range: {mat_file: {background: [list of range], cpa: [list of range]}}
    label_range_file = "/home/sl29/data/ACIDS/mat_label_range.json"
    with open(label_range_file, "r") as f:
        label_range = json.load(f)

    # 126 qualified files
    files_to_process = []
    files_to_skip = {"Gv2b2198.mat", "Gv7a1068.mat"}
    for e in os.listdir(os.path.join(input_path, "Acoustics")):
        if e.endswith(".mat") and e in meta_info and e in label_range and e not in files_to_skip:
            files_to_process.append(e[:-4])
    print(f"Total number of files to process: {len(files_to_process)}")

    # partition the data
    if parition_option == "random":
        train_files, val_files, test_files = random_parition_files(files_to_process)
    else:
        train_files, val_files, test_files = TRAIN_FILES, VAL_FILES, TEST_FILES

    for task in [
        "vehicle_classification",
        "speed_classification",
        "distance_classification",
        "terrain_classification",
    ]:
        print("=" * 80)
        print(f"Partitioning data for task: {task}.")
        output_path = f"/home/sl29/data/ACIDS/{parition_option}_partition_index_{task}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # partition the dta
        partition_data(task, paired_data_path, output_path, train_files, val_files, test_files)
