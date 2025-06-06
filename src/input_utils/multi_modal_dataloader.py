import os
import torch
import logging
import random
import numpy as np

from torch.utils.data import DataLoader
from input_utils.multi_modal_dataset import MultiModalDataset, MultiModalSequenceDataset
from torch.utils.data import Sampler


def create_dataloader(dataloader_option, args, batch_size=64, workers=5):
    """create the dataloader for the given data path.

    Args:
        option (_type_): training, validation, testing dataset
        data_path (_type_): _description_
        workers (_type_): _description_
    """
    # select the index file
    label_ratio = 1
    if dataloader_option == "train":
        if args.train_mode not in {"supervised"} and args.stage == "pretrain":
            """self-supervised training"""
            index_file = args.dataset_config["pretrain_index_file"]
        else:
            """supervised training"""
            index_file = args.dataset_config[args.task]["train_index_file"]
            label_ratio = args.label_ratio
    elif dataloader_option == "val":
        index_file = args.dataset_config[args.task]["val_index_file"]
    else:
        index_file = args.dataset_config[args.task]["test_index_file"]

    # init the flags
    balanced_sample_flag = (
        args.balanced_sample
        and args.option == "train"
        and dataloader_option == "train"
        and (args.train_mode == "supervised" or args.stage == "finetune")
    )
    sequence_sampler_flag = args.sequence_sampler and args.train_mode == "contrastive" and args.stage == "pretrain"

    # init the dataset
    if sequence_sampler_flag:
        dataset = MultiModalSequenceDataset(args, index_file)
        batch_size = min(batch_size, len(dataset) * args.dataset_config["seq_len"])
    else:
        dataset = MultiModalDataset(args, index_file, label_ratio, balanced_sample_flag)
        batch_size = min(batch_size, len(dataset))

    # define the dataloader with weighted sampler for training
    if balanced_sample_flag:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.sample_weights, dataset.epoch_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
        logging.info("=\tUsing class balanced sampler for training")
    elif sequence_sampler_flag:
        sampler = BatchSeqSampler(args, batch_size, dataset)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers)
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(dataloader_option == "train"), num_workers=workers
        )

    return dataloader


class BatchSeqSampler(Sampler):
    def __init__(self, args, batch_size, dataset):
        self.dataset = dataset
        self.args = args
        self.subseq_batch_size = batch_size // args.dataset_config["seq_len"]
        self.subseq_count = len(self.dataset.subseqs)
        self.subseq_indices = list(range(self.subseq_count))

    def __iter__(self):
        """
        Randomly sample a subset of subsequences, and return the list of contained sample indices.
        """
        random.shuffle(self.subseq_indices)

        # trunk data and concat samples of subsequences
        for batch_id in range(0, self.subseq_count, self.subseq_batch_size):
            batch_sample_indices = []
            batch_subseqs = [
                self.dataset.subseqs[subseq_id]
                for subseq_id in self.subseq_indices[batch_id : batch_id + self.subseq_batch_size]
            ]
            for subseq in batch_subseqs:
                batch_sample_indices.extend(self.dataset.subseq_to_sample_idx[subseq])

            yield batch_sample_indices

    def __len__(self):
        return np.ceil(self.subseq_count / self.subseq_batch_size).astype(int)


def preprocess_triplet_batch(data, labels):
    """Preprocess the triplet batch by concatenating all elements within the tuple (anchor, pos, neg)

    Args:
        flag (_type_): _description_
        data (_type_): _description_
        labels (_type_): _description_
    """
    # cat data
    anchor_data, pos_data, neg_data = data
    out_data = dict()
    for loc in anchor_data:
        out_data[loc] = dict()
        for mod in anchor_data[loc]:
            out_data[loc][mod] = torch.cat([anchor_data[loc][mod], pos_data[loc][mod], neg_data[loc][mod]], dim=0)

    # cat labels
    out_labels = torch.cat(labels, dim=0)

    return out_data, out_labels