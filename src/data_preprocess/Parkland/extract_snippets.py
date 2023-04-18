import os
import torch
import random
import getpass

import numpy as np

from tqdm import tqdm


def convert_samples(input_path, output_path):
    """
    For each sample, we use 10% prob to keep and convert it to shared format.
    data: {
        audio: [i*s]
        label: float,
    }
    """
    for sample in tqdm(os.listdir(input_path)):
        sample_path = os.path.join(input_path, sample)
        out_sample_path = os.path.join(output_path, sample)

        if random.random() < 0.1:
            sample_data = torch.load(sample_path)
            out_sample_data = {}

            # convert data
            out_sample_data["audio"] = sample_data["data"]["shake"]["audio"].flatten()
            out_sample_data["label"] = sample_data["label"]

            # write the out file
            torch.save(out_sample_data, out_sample_path)


if __name__ == "__main__":
    noisy_sample_path = "/home/sl29/data/Parkland_1107/individual_noisy_time_samples"
    clean_sample_path = "/home/sl29/data/Parkland_1107/individual_clean_time_samples"
    output_path = "/home/sl29/data/Parkland_1107/samples_share"

    convert_samples(noisy_sample_path, output_path)
    convert_samples(clean_sample_path, output_path)
