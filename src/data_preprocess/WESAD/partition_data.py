import os
import torch
import random
import getpass

import numpy as np

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


def partition_data(paired_data_path, output_path, user_list, train_ratio=0.8):
    """Partition the data according to the given ratio, using all-but-one strategy.
    We don't touch the processed data, but only save a new index file.

    Args:
        paired_data_path (_type_): _description_
        output_path (_type_): _description_
        train_ratio (_type_): _description_
    """
    # partition the user list into training and testing
    random.shuffle(user_list)
    train_user_count = 11
    val_user_count = 2
    train_users = set(user_list[:train_user_count])
    val_users = set(user_list[train_user_count : train_user_count + val_user_count])
    test_users = set(user_list[train_user_count + val_user_count :])
    print(
        f"Number of training users: {len(train_users)}, \
        number of validation users: {len(val_users)}, \
        number of testing users: {len(test_users)}."
    )

    # for users in training set, only preserve their data samples with complete modalities
    data_samples = os.listdir(paired_data_path)
    train_samples = []
    val_samples = []
    test_samples = []

    for sample in tqdm(data_samples):
        user = os.path.basename(sample).split("_")[0]
        file_path = os.path.join(os.path.join(paired_data_path, sample))

        if user in test_users:
            target = test_samples
        elif user in val_users:
            target = val_samples
        else:
            target = train_samples

        """For all users, we only preserve samples with complete modalities in the dataset."""
        load_sample = torch.load(file_path)
        complete_modality_flag = 1
        for loc in load_sample["flag"]:
            for mod in load_sample["flag"][loc]:
                complete_modality_flag *= load_sample["flag"][loc][mod]

        if complete_modality_flag:
            target.append(file_path)

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
<<<<<<< HEAD
    paired_data_path = f"/home/{username}/data/WESAD/time_individual_samples"
    output_path = f"/home/{username}/data/WESAD/time_data_partition"
=======
    paired_data_path = f"/home/{username}/data/WESAD/time_individual_samples_four_class"
    output_path = f"/home/{username}/data/WESAD/time_data_partition_four_class"
>>>>>>> 7fbfff994bafea966da52abdb6f33c38f5146425

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # extract user list
    users = extract_user_list(f"/home/{username}/data/WESAD/raw_data/WESAD")

    # partition the dta
    partition_data(paired_data_path, output_path, users)
