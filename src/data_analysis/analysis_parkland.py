import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
DATA_FILE = "/home/tkimura4/data/datasets/MOD/Parkland/time_data_partition/train_index.txt"
DATA_FILE_ACIDS = "/home/tkimura4/data/datasets/ACIDS/random_partition_index_vehicle_classification/train_index.txt"

DATA_FILE = DATA_FILE_ACIDS

CLASS_NAMES = ["Polaris", "Warhog", "Truck", "motor", "tesla", "mustang", "walk"]


# Load the file names from DATA_FILE
with open(DATA_FILE, "r") as f:
    file_names = f.readlines()

file_names = [file_name.strip() for file_name in file_names]

file_names = file_names[:300]

# Load each sample from the file names 
all_samples = []
for file_name in file_names:
    sample = torch.load(file_name)
    all_samples.append({"audio": sample['data']['shake']['audio'], "seismic": sample['data']['shake']['seismic']})

all_audio = [sample['audio'] for sample in all_samples]
all_seismic = [sample['seismic'] for sample in all_samples]

all_audio = np.concatenate(all_audio, axis=0)
all_seismic = np.concatenate(all_seismic, axis=0)

print(all_audio.shape, all_seismic.shape)

# plot the histogram of the audio and seismic data
plt.hist(all_audio.flatten(), bins=100)
plt.savefig("audio_histogram.png")
plt.close()

plt.hist(all_seismic.flatten(), bins=100)
plt.savefig("seismic_histogram.png")
plt.close()

# analyze the data summary for sample['data']['shake']['seismic'] and sample['data']['shake']['audio']



