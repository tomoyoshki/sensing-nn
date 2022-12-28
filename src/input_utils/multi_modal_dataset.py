import os
import torch
import numpy as np

from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, index_file, base_sample_path):
        '''_summary_

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
        '''
        self.sample_files = list(np.loadtxt(os.path.join(base_sample_path, index_file), dtype=str))
        self.base_sample_path = base_sample_path

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_file = os.path.join(self.base_sample_path, self.sample_files[idx])
        sample = torch.load(sample_file)
        label = sample["label"]
        data = sample["data"]

        return data, label


class TripletMultiModalDataset(Dataset):
    def __init__(self, index_file, base_sample_path):
        '''Reference:
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
        '''
        self.sample_files = list(np.loadtxt(os.path.join(base_sample_path, index_file), dtype=str))
        self.base_sample_path = base_sample_path

        # extract the label to id mapping
        self.label_to_ids = dict()
        for i, sample_file in enumerate(self.sample_files):
            sample_file = os.path.join(self.base_sample_path, self.sample_files[i])
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
        '''Read the sample at the given id.

        Args:
            idx (_type_): _description_
        '''
        sample_file = os.path.join(self.base_sample_path, self.sample_files[idx])
        sample = torch.load(sample_file)
        label = sample["label"]
        # flag = sample["flag"]
        data = sample["data"]

        return data, label
