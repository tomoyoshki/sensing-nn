import os
import torch
import logging
import numpy as np

from torch.utils.data import Dataset
from random import shuffle


class MultiModalDataset(Dataset):
    def __init__(self, args, index_file, label_ratio=1, balanced_sample=False):
        self.args = args
        self.index_file = index_file
        self.sample_files = list(np.loadtxt(index_file, dtype=str))

        self.label_dict = {
            "gle350": 0,
            "miata": 1,
            "cx30": 2,
            "mustang": 3,
        }
        
        print(len(self.sample_files))

        if label_ratio < 1:
            shuffle(self.sample_files)
            self.sample_files = self.sample_files[: round(len(self.sample_files) * label_ratio)]

        if balanced_sample:
            self.load_sample_labels()

    def load_sample_labels(self):
        logging.info(f"=\tBalancing samples")
        sample_labels = []
        label_count = [0 for i in range(self.args.num_class)]

        for idx in range(len(self.sample_files)):
            _, label, detection_label, _ = self.__getitem__(idx)

            if label.shape[0] > 1:
                label = torch.argmax(label).item()
                sample_labels.append(label)
                label_count[label] += 1
            else:
                sample_labels.append(label)
                label_count[label] += 1
        self.sample_weights = []
        self.epoch_len = int(np.max(label_count) * len(label_count))
        for sample_label in sample_labels:
            self.sample_weights.append(1 / label_count[sample_label])

    def __len__(self):
        return len(self.sample_files)
    
    def get_normal(self, sample, pt_file, idx):
        data = sample["data"]
        # ACIDS and Parkland
        if isinstance(sample["label"], dict):
            if self.args.task == "vehicle_classification":
                label = sample["label"]["vehicle_type"]
            elif self.args.task == "distance_classification":
                label = sample["label"]["distance"]
            elif self.args.task == "speed_classification":
                label = sample["label"]["speed"] // 5 - 1
            elif self.args.task == "terrain_classification":
                label = sample["label"]["terrain"]
            else:
                raise ValueError(f"Unknown task: {self.args.task}")
        else:
            label = sample["label"]

        if isinstance(label, str):
            if label not in self.label_dict:
                print(f"Label not in the dictionary: {label}")
            label = self.label_dict[label]

        dist = int(pt_file.split(".")[0].split("_")[-1])

        dist_threshold = 15

        if ("multiclass" in self.args.finetune_tag or "dist" in self.args.finetune_tag) and dist > dist_threshold:
            label = 4  # background

        # if "detection" in self.args.finetune_tag:

        detection_label = -1

        if dist > dist_threshold:
            detection_label = 0  # no car
        else:
            detection_label = 1  # has car

        for loc in data:
            for mod in data[loc]:
                if data[loc][mod].ndim == 2:
                    data[loc][mod] = torch.from_numpy(data[loc][mod]).unsqueeze(0)

        return data, label, detection_label, idx

    def get_ict(self, data):
        label = data["vehicle_id"]
        seismic_data = data["seismic"]
        acousitc_data = data["acoustic"]

        dict_data = {
            "shake": {"seismic": seismic_data[::2].reshape(1, 10, 20), "audio": acousitc_data[::2].reshape(1, 10, 1600)}
        }

        detection_label = torch.tensor(0)
        if data["meta/distance_mean"] is not None and data["meta/distance_mean"] <= 15:
            detection_label = torch.tensor(1)  # has car

        return dict_data, label, detection_label, 0

    def get_ict_dual(self, data):
        vehicles = data["vehicle_id"].item()
        if isinstance(vehicles, int):
            vehicles = [vehicles]
        
        labels = torch.FloatTensor(vehicles)

        seismic_data = data["seismic"]
        acousitc_data = data["acoustic"]

        dict_data = {
            "shake": {"seismic": seismic_data[::2].reshape(1, 10, 20), "audio": acousitc_data[::2].reshape(1, 10, 1600)}
        }

        detection_label = torch.tensor(0)

        if data["dis"] is not None:
            dis_mean = data["dis"].mean(dim=0)  # mean distance
            for dis in data["dis"]:
                if dis is not None and dis <= 15:
                    detection_label = torch.tensor(1)  # has car if one of the vehicle is present
        else:
            dis_mean = torch.tensor(-1)

        meta_data = dis_mean

        if self.args.multi_class:
            if detection_label > 0:
                # has car
                labels = torch.nn.functional.one_hot(labels, num_classes=self.args.num_class)
            else:
                labels = torch.LongTensor([0] * self.args.num_class)

        detection_label = detection_label.reshape(1)
        return dict_data, labels, detection_label.float(), meta_data.float()

    def get_gcq(self, sample):
        data = sample["data"]
        detection_label = torch.tensor(1)

        if sample["label"] == "background":
            detection_label = torch.tensor(0)

        label = sample["label"]

        label_dict = {"polaris": 0, "silverado": 1, "sedan": 1, "warthog": 2, "husky": 3, "background": -1}
        if isinstance(label, str):
            label = torch.LongTensor([label_dict[label]])

        label = label.long().item()

        for loc in data:
            for mod in data[loc]:
                data[loc][mod] = data[loc][mod].reshape(1, 10, -1)
                if isinstance(data[loc][mod], np.ndarray):
                    data[loc][mod] = torch.from_numpy(data[loc][mod])
                data[loc][mod] = data[loc][mod].float()
        meta_data = -1

        return data, label, detection_label, meta_data

    def get_gcq_mixed(self, sample):
        data = sample["data"]
        detection_label = torch.tensor(1)

        if sample["label"] == "background":
            detection_label = torch.tensor(0)

        label = sample["label"]

        label_dict = {"polaris": 0, "silverado": 1, "sedan": 1, "warthog": 2, "husky": 3, "background": -1}
        if isinstance(label, str):
            label = torch.LongTensor([label_dict[label]])

        if not isinstance(label, list):
            label = [label]

        multi_label = torch.FloatTensor([0] * self.args.num_class)
        if detection_label > 0:
            for l in label:
                if l >= 0:
                    multi_label[l] = 1

        for loc in data:
            for mod in data[loc]:
                data[loc][mod] = data[loc][mod].reshape(1, 10, -1)
                if isinstance(data[loc][mod], np.ndarray):
                    data[loc][mod] = torch.from_numpy(data[loc][mod])
                data[loc][mod] = data[loc][mod].float()
        meta_data = -1
        detection_label = detection_label.reshape(1)
        return data, multi_label, detection_label, meta_data

    def get_ict_multi(self, data):
        
        if "dual" in self.args.finetune_tag:
            return self.get_ict_dual(data)

        vehicles = data["vehicle_id"].item()

        if isinstance(vehicles, int):
            vehicles = [vehicles]

        multi_label = torch.zeros(4)
        for vid in vehicles:
            multi_label[vid] += 1

        seismic_data = data["seismic"]
        acousitc_data = data["acoustic"]

        dict_data = {
            "shake": {"seismic": seismic_data[::2].reshape(1, 10, 20), "audio": acousitc_data[::2].reshape(1, 10, 1600)}
        }

        detection_label = torch.tensor(0)

        if data["dis"] is not None:
            dis_mean = data["dis"].mean(dim=0)  # mean distance
            for dis in data["dis"]:
                if dis is not None and dis <= 15:
                    detection_label = torch.tensor(1)  # has car if one of the vehicle is present
        else:
            dis_mean = torch.tensor(100)

        if detection_label == 0 and self.args.option != "test":
            multi_label = torch.FloatTensor([0] * 4)

        return dict_data, multi_label, detection_label.float(), dis_mean.float()

    def get_gcq_aug(self, sample):
        data = sample["data"]
        label = sample["label"]
        distance = sample["distance"]
        label_dict = {"polaris": 0, "silverado": 1, "sedan": 1, "truck": 1, "warthog": 2, "background": -1}
        class_labels = torch.FloatTensor([0] * 3)
        for l in label:
            class_labels[label_dict[l]] = 1
            distance_label = torch.FloatTensor([distance[l]])
        
        
        if self.args.task == "vehicle_classification":
            label = class_labels
        elif self.args.task == "distance_regression":
            label = distance_label
        elif self.args.task == "distance_classification":
            if distance_label < 15:
                label = torch.FloatTensor([0])
            elif distance_label < 30:
                label = torch.FloatTensor([1])
            else:
                label = torch.FloatTensor([2])
        return data, label, 0
    
    def __getitem__(self, idx):
        pt_file = self.sample_files[idx]
        sample = torch.load(pt_file)

        for tag in ["ictexclusive"]:
            if self.args.option == "train" and tag in self.args.finetune_set:
                return self.get_ict_multi(sample)
            if self.args.option == "test" and tag in self.args.test_set:
                return self.get_ict_multi(sample)

        for tag in ["ictfiltered", "ictyizhuo"]:
            if self.args.option == "train" and tag in self.args.finetune_set:
                return self.get_ict(sample)
            if self.args.option == "test" and tag in self.args.test_set:
                return self.get_ict(sample)


        for tag in ["gcqallmixed", "gcqday1filtered", "gcqday2filtered", "gcqmixed"]:
            if self.args.option == "train" and tag in self.args.finetune_set:
                return self.get_gcq_mixed(sample)
            if self.args.option == "test" and tag in self.args.test_set:
                return self.get_gcq_mixed(sample)

        for tag in ["gcqall"]:
            if self.args.option == "train" and tag in self.args.finetune_set:
                return self.get_gcq(sample)
            if self.args.option == "test" and tag in self.args.test_set:
                return self.get_gcq(sample) 
            
        for tag in ["gcq20240806", "gcq20240807"]:
            if self.args.option == "train" and tag in self.args.finetune_set:
                return self.get_gcq_aug(sample)
            if self.args.option == "test" and tag in self.args.test_set:
                return self.get_gcq_aug(sample)         
        
        return self.get_normal(sample, pt_file, idx)


class MultiModalSequenceDataset(Dataset):
    def __init__(self, args, index_file):
        """
        Extract multiple sequences of consecutive samples at the time dimension.
        """
        self.args = args
        self.sample_files = list(np.loadtxt(index_file, dtype=str))
        self.partition_subsequences()

    def partition_subsequences(self):
        """
        Extract all sequence IDs from the sample files.
        seq_to_sample: {sequence_id: [(sample_id, sample_file), ...], ...}
        """
        seq_len = self.args.dataset_config["seq_len"]

        if self.args.dataset == "RealWorld_HAR":
            delimiter = "-"
        else:
            delimiter = "_"

        seq_to_samples = {}
        for sample_idx, sample_file in enumerate(self.sample_files):
            # Sequence ID is separeted by the last underscore symbol.
            basename = os.path.basename(sample_file)
            seq = basename.rsplit(delimiter, 1)[0]

            if seq not in seq_to_samples:
                seq_to_samples[seq] = [(sample_idx, sample_file)]
            else:
                seq_to_samples[seq].append((sample_idx, sample_file))

        # sort the sequences
        for seq in seq_to_samples:
            seq_to_samples[seq].sort(key=lambda x: int(os.path.basename(x[1]).rsplit(delimiter, 1)[1].split(".")[0]))
            seq_to_samples[seq] = [e[0] for e in seq_to_samples[seq]]

        # divide sequences into subsequences of fixed length
        self.subseqs = []
        self.subseq_to_sample_idx = {}
        for seq in seq_to_samples:
            for i in range(0, len(seq_to_samples[seq]), seq_len):
                subseq = f"{seq}_{i}"
                self.subseqs.append(subseq)

                # constitute the sample list with fixed length
                sample_id_list = seq_to_samples[seq][i : i + seq_len]
                while len(sample_id_list) < seq_len:
                    sample_id_list.append(sample_id_list[-1])

                self.subseq_to_sample_idx[subseq] = sample_id_list

    def __len__(self):
        return len(self.subseqs)

    def __getitem__(self, sample_idx):
        """
        Extract a random sequence of samples.
        """
        sample = torch.load(self.sample_files[sample_idx])
        data = sample["data"]

        # ACIDS
        if isinstance(sample["label"], dict):
            if self.args.task == "vehicle_classification":
                label = sample["label"]["vehicle_type"]
            elif self.args.task == "distance_classification":
                label = sample["label"]["distance"]
            elif self.args.task == "speed_classification":
                label = sample["label"]["speed"] // 5 - 1
            elif self.args.task == "terrain_classification":
                label = sample["label"]["terrain"]
            else:
                raise ValueError(f"Unknown task: {self.args.task}")
        else:
            label = sample["label"]

        self.label_dict = {
            "gle350": 7,
            "miata": 8,
            "cx30": 9,
            "mustang": 5,
        }

        if isinstance(label, str):
            if label not in self.label_dict:
                print(f"Label not in the dictionary: {label}")
            label = self.label_dict[label]
            label = torch.LongTensor([label]).long()

        label = label.float()
        label = label.unsqueeze(0).reshape(1)
        for loc in data:
            for mod in data[loc]:
                if data[loc][mod].ndim == 2:
                    data[loc][mod] = torch.from_numpy(data[loc][mod]).unsqueeze(0).float()

        return data, label, sample_idx


class TripletMultiModalDataset(Dataset):
    def __init__(self, index_file, base_path):
        """
        Reference:
            https://github.com/adambielski/siamese-triplet/blob/0c719f9e8f59fa386e8c59d10b2ddde9fac46276/datasets.py#L79
        """
        self.sample_files = list(np.loadtxt(os.path.join(base_path, index_file), dtype=str))
        self.base_path = base_path

        # extract the label to id mapping
        self.label_to_ids = dict()
        for i, sample_file in enumerate(self.sample_files):
            sample_file = os.path.join(self.base_path, self.sample_files[i])
            label = torch.load(sample_file)["label"].item()
            if label not in self.label_to_ids:
                self.label_to_ids[label] = [i]
            else:
                self.label_to_ids[label].append(i)
        self.label_set = set(self.label_to_ids.keys())

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # the anchor sample
        anchor_data, anchor_label = self.read_one_sample(idx)
        anchor_label_val = anchor_label.item()

        # randomly select the positive sample
        pos_id = idx
        while pos_id == idx:
            pos_id = np.random.choice(self.label_to_ids[anchor_label_val])
        pos_data, pos_label = self.read_one_sample(pos_id)

        # randomly select the negative sample
        neg_label = np.random.choice(list(self.label_set - set([anchor_label_val])))
        neg_id = np.random.choice(self.label_to_ids[neg_label])
        neg_data, neg_label = self.read_one_sample(neg_id)

        return (
            (anchor_data, pos_data, neg_data),
            (anchor_label, pos_label, neg_label),
        )

    def read_one_sample(self, idx):
        """
        Read the sample at the given id.

        Args:
            idx (_type_): _description_
        """
        sample_file = os.path.join(self.base_path, self.sample_files[idx])
        sample = torch.load(sample_file)
        label = sample["label"]
        data = sample["data"]

        return data, label
