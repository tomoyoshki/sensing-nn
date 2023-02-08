import os
import torch
import numpy as np

from torch.utils.data import Dataset
from random import shuffle


class MultiModalDataset(Dataset):
    def __init__(self, args, index_file, label_ratio=1, balanced_sample=False):
        """
        Args:
            modalities (_type_): The list of modalities
            classes (_type_): The list of classes
            index_file (_type_): The list of sample file names
            sample_path (_type_): The base sample path.

        Sample:
            - label: Tensor
            - flag
                - phone
                    - audio: True
                    - acc: False
            - data:
                -phone
                    - audio: Tensor
                    - acc: Tensor
        """
        self.args = args
        self.sample_files = list(np.loadtxt(index_file, dtype=str))

        if label_ratio < 1:
            shuffle(self.sample_files)
            self.sample_files = self.sample_files[: round(len(self.sample_files) * label_ratio)]

        if balanced_sample:
            self.load_sample_labels()

    def load_sample_labels(self):
        sample_labels = []
        label_count = [0 for i in range(self.args.dataset_config[self.args.task]["num_classes"])]

        for idx in range(len(self.sample_files)):
            _, label = self.__getitem__(idx)
            label = torch.argmax(label).item() if label.numel() > 1 else label.item()
            sample_labels.append(label)
            label_count[label] += 1

        self.sample_weights = []
        self.epoch_len = int(np.max(label_count) * len(label_count))
        for sample_label in sample_labels:
            self.sample_weights.append(1 / label_count[sample_label])

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_files[idx])
        data = sample["data"]

        # ACIDS
        if isinstance(sample["label"], dict):
            if self.args.task == "vehicle_classification":
                label = sample["label"]["vehicle_type"]
            elif self.args.task == "distance_classification":
                label = sample["label"]["distance"]
            elif self.args.task == "speed_classification":
                label = sample["label"]["speed"]
            elif self.args.task == "terrain_classification":
                label = sample["label"]["terrain"]
            else:
                raise ValueError(f"Unknown task: {self.args.task}")
        else:
            label = sample["label"]

        return data, label, idx


class TripletMultiModalDataset(Dataset):
    def __init__(self, index_file, base_path):
        """
        Reference:
            https://github.com/adambielski/siamese-triplet/blob/0c719f9e8f59fa386e8c59d10b2ddde9fac46276/datasets.py#L79

        Args:
            modalities (_type_): The list of modalities
            classes (_type_): The list of classes
            index_file (_type_): The list of sample file names
            sample_path (_type_): The base sample path.

        Sample:
            - label: Tensor
            - flag
                - phone
                    - audio: True
                    - acc: False
            - data:
                -phone
                    - audio: Tensor
                    - acc: Tensor

        Function:
            Generate a triplet of samples (anchor, pos, neg) within each batch
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
