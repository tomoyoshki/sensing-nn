import numpy as np
from os.path import exists
import os
import scipy.io as scio
from meta_loader import load_meta
from scipy import signal

PATH_A = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat/Acoustics"
PATH_S = "/home/sl29/data/ACIDS/ACIDSData_public_testset-mat/Seismic"
TEST_FILE = "/home/sl29/FoundationSense/src/data_preprocess/ACIDS/test_file_list.txt"

FILES = [
    "Gv1a1002.mat",
    "Gv1a2002.mat",
    "Gv1a1012.mat",
    "Gv1a2012.mat",
    "Gv1b1020.mat",
    "Gv1b2020.mat",
    "Gv1b1164.mat",
    "Gv1b2164.mat",
    "Gv1b2022.mat",
    "Gv1c1020.mat",
    "Gv1c2020.mat",
    "Gv1c1030.mat",
    "Gv1c2030.mat",
    "Gv1c1084.mat",
    "Gv1c2022.mat",
    "Gv1d1132.mat",
    "Gv2a1078.mat",
    "Gv2a1080.mat",
    "Gv2b1008.mat",
    "Gv2b2008.mat",
    "Gv2c1008.mat",
    "Gv2c1010.mat",
    "Gv2c2008.mat",
    "Gv2c2010.mat",
    "Gv3c1034.mat",
    "Gv3c1036.mat",
    "Gv4c1040.mat",
    "Gv4c1042.mat",
    "Gv4c1044.mat",
    "Gv4c1096.mat",
    "Gv4d1036.mat",
    "Gv4d2036.mat",
    "Gv5a1046.mat",
    "Gv5a2046.mat",
    "Gv5a1048.mat",
    "Gv5a2048.mat",
    "Gv5a1108.mat",
    "Gv5b2046.mat",
    "Gv5c1046.mat",
    "Gv5c2046.mat",
    "Gv5c1102.mat",
    "Gv5d1120.mat",
    "Gv6c1050.mat",
    "Gv6c2050.mat",
    "Gv6c1052.mat",
    "Gv6d1014.mat",
    "Gv6d2014.mat",
    "Gv6d1016.mat",
    "Gv6d2016.mat",
    "Gv6d1018.mat",
    "Gv6d2018.mat",
    "Gv6d1086.mat",
    "Gv6d2086.mat",
    "Gv7a1068.mat",
    "Gv7a1130.mat",
    "Gv8a1056.mat",
    "Gv8b1058.mat",
    "Gv8c1062.mat",
    "Gv8c2062.mat",
    "Gv8c1118.mat",
    "Gv8c1210.mat",
    "Gv8c1212.mat",
    "Gv8d1108.mat",
    "Gv9a1060.mat",
    "Gv9c1070.mat",
    "Gv9c2070.mat",
    "Gv9c1126.mat",
    "Gv9d1126.mat",
    "Gv1a1136.mat",
    "Gv1a2014.mat",
    "Gv1b1166.mat",
    "Gv1b2166.mat",
    "Gv1c1086.mat",
    "Gv1c1202.mat",
    "Gv1d1134.mat",
    "Gv2a1052.mat",
    "Gv2a2052.mat",
    "Gv2b1010.mat",
    "Gv2b2010.mat",
    "Gv2c1080.mat",
    "Gv3c1090.mat",
    "Gv4c1098.mat",
    "Gv4c1100.mat",
    "Gv4d1038.mat",
    "Gv4d2038.mat",
    "Gv5a1148.mat",
    "Gv5a2148.mat",
    "Gv5b2048.mat",
    "Gv5c1048.mat",
    "Gv5c2048.mat",
    "Gv6c2052.mat",
    "Gv6d1088.mat",
    "Gv6d2088.mat",
    "Gv7c1056.mat",
    "Gv8a2056.mat",
    "Gv8b2058.mat",
    "Gv8c1214.mat",
    "Gv8c1216.mat",
    "Gv8d1110.mat",
    "Gv9a1122.mat",
    "Gv9c1128.mat",
    "Gv1a1004.mat",
    "Gv1a2134.mat",
    "Gv1b2168.mat",
    "Gv1b2170.mat",
    "Gv1c1032.mat",
    "Gv1c1204.mat",
    "Gv1c2032.mat",
    "Gv1d1136.mat",
    "Gv2a1140.mat",
    "Gv2a2140.mat",
    "Gv2b1198.mat",
    "Gv2b2198.mat",
    "Gv2c1082.mat",
    "Gv3c1092.mat",
    "Gv4c1206.mat",
    "Gv4c1208.mat",
    "Gv4d1040.mat",
    "Gv4d2040.mat",
    "Gv5a1150.mat",
    "Gv5a2150.mat",
    "Gv5c1104.mat",
    "Gv5d1122.mat",
    "Gv6c1106.mat",
    "Gv6d1090.mat",
    "Gv6d2090.mat",
    "Gv7c1058.mat",
    "Gv8a2060.mat",
    "Gv8b2060.mat",
    "Gv8c1218.mat",
    "Gv8c1220.mat",
    "Gv8d1112.mat",
    "Gv9c1072.mat",
    "Gv9c2072.mat",
    "Gv9d1128.mat",
]

SAMPLE_LEN = 1024


def gene_data(FILES_SET, mode="fft"):
    meta_info = load_meta()
    X = []
    Y = []
    file_sample_count = np.zeros(len(FILES_SET))
    file_labels = []
    for n, filename in enumerate(FILES_SET):
        label = int(filename[2])

        if filename not in meta_info:
            continue

        y = meta_info[filename]

        file_labels.append(label)
        if not exists(os.path.join(PATH_A, filename)):
            print("A File not exist: " + filename)
            continue

        s_filename = filename[0:3] + "s" + filename[3:]
        if not exists(os.path.join(PATH_S, s_filename)):
            print("S File not exist: " + s_filename)
            continue

        data = scio.loadmat(os.path.join(PATH_A, filename))
        x_a = data["Output_data"]
        x_a = x_a[:, :-20]  # Delete outliers

        data = scio.loadmat(os.path.join(PATH_S, s_filename))
        x_s = data["Output_data"]
        x_s = x_s[:, :-20]  # Delete outliers

        if x_a.shape[1] != x_s.shape[1]:
            print("shape not equal", x_a.shape, x_s.shape)

        x = np.concatenate((x_a, x_s), axis=0)
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)

        i = 0
        while i + SAMPLE_LEN <= x.shape[1]:
            file_sample_count[n] += 1

            if mode == "fft":
                fft_signal = np.fft.fft(x[:, i : i + SAMPLE_LEN], axis=-1)[:, : SAMPLE_LEN // 2]
                fft_signal = np.concatenate([fft_signal.real, fft_signal.imag], axis=0)
                fft_signal = np.concatenate(
                    [[fft_signal[0]], [fft_signal[5]], [fft_signal[3]], [fft_signal[8]]], axis=0
                )  # Keep the real and imag parts of sensor 0 and 3

                X.append(fft_signal)
                Y.append(y)
            elif mode == "stft":
                f, t, Zxx = signal.stft(x[:, i : i + SAMPLE_LEN], 1024, nperseg=128, noverlap=64)
                stft_signal = np.abs(Zxx)
                stft_signal = np.transpose(stft_signal, (0, 2, 1))

                # Only take 1 axis from each sensor
                stft_signal = np.concatenate([[stft_signal[0]], [stft_signal[3]]], axis=0)
                X.append(stft_signal)
                Y.append(y)
            # item = x[:,i:i+SAMPLE_LEN]
            # x_dict[label].append(item)
            i += SAMPLE_LEN

    X = np.array(X)
    # if mode == "fft":
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.array(X), np.array(Y), file_sample_count, file_labels


def load_data(mode="fft"):
    test_files = []
    with open(TEST_FILE, "r") as f:
        for line in f.readlines():
            test_files.append(line.strip())

    train_files = []
    for fn in FILES:
        if fn not in test_files:
            train_files.append(fn)
    train_X, train_Y, train_sample_count, train_labels = gene_data(train_files, mode=mode)
    test_X, test_Y, test_sample_count, test_labels = gene_data(test_files, mode=mode)

    return train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels
