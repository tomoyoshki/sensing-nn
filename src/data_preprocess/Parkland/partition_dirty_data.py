from cgi import test
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


def partition_data(paired_data_path, output_path):
    """Partition the data according to the given ratio, using all-but-one strategy.
    We don't touch the processed data, but only save a new index file.

    Args:
        paired_data_path (_type_): _description_
        output_path (_type_): _description_
        train_ratio (_type_): _description_
    """
    # for users in training set, only preserve their data samples with complete modalities
    data_samples = os.listdir(paired_data_path)
    test_samples = []

    for sample in tqdm(data_samples):
        file_path = os.path.join(os.path.join(paired_data_path, sample))
        target = test_samples

        """For all users, we only preserve samples with complete modalities in the dataset."""
        load_sample = torch.load(file_path)
        complete_modality_flag = 1
        for loc in load_sample["flag"]:
            for mod in load_sample["flag"][loc]:
                complete_modality_flag *= load_sample["flag"][loc][mod]

        if complete_modality_flag:
            target.append(file_path)

    # save the index file
    print(f"number of testing samples: {len(test_samples)}.")
    if test_1107:
        if clean:
            index_file = os.path.join(output_path, "test_index_1107_clean.txt")
        else:
            index_file = os.path.join(output_path, "test_index_1107_noisy.txt")
    else:
        index_file = os.path.join(output_path, "noisy_test_index.txt")

    with open(os.path.join(output_path, index_file), "w") as f:
        for sample_file in test_samples:
            f.write(sample_file + "\n")

    # Synchronize the data from INCAS --> Eugene
    cmd = f"rsync -av {output_path}/ eugene:{output_path}/"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    username = getpass.getuser()
    test_1107 = True
    clean = False

    if test_1107:
        if clean:
            paired_data_path = f"/home/{username}/data/Parkland_1107/individual_clean_time_samples"
        else:
            paired_data_path = f"/home/{username}/data/Parkland_1107/individual_noisy_time_samples"
    else:
        paired_data_path = f"/home/{username}/data/Parkland_Dirty/individual_time_samples"

    output_path = f"/home/{username}/data/Parkland/time_data_partition"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # partition the dta
    partition_data(paired_data_path, output_path)
