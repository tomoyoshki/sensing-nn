from cgi import test
import os
import torch
import random
import getpass

import numpy as np

from tqdm import tqdm


def partition_data(paired_data_path, output_path, train_ratio=0.9):
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
        file_path = os.path.join(os.path.join(paired_data_path, sample))

        if random.random() < 0.9:
            continue

        if random.random() < train_ratio:
            target = train_samples
        else:
            target = test_samples

        """For all users, we only preserve samples with complete modalities in the dataset."""
        load_sample = torch.load(file_path)
        complete_modality_flag = 1
        for loc in load_sample["flag"]:
            for mod in load_sample["flag"][loc]:
                complete_modality_flag *= load_sample["flag"][loc][mod]

        if complete_modality_flag:
            target.append(file_path)

    # same val as test examples
    val_samples = test_samples

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
    username = getpass.getuser()

    paired_data_path = f"/home/{username}/data/2023-03-27-State-Farm-Center-SW-Lot/individual_time_samples"
    output_path = f"/home/{username}/data/2023-03-27-State-Farm-Center-SW-Lot/time_data_partition"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # partition the dta
    partition_data(paired_data_path, output_path)
