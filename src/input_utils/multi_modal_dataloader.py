import os
import torch
import logging
import numpy as np

from torch.utils.data import DataLoader
from input_utils.multi_modal_dataset import MultiModalDataset, TripletMultiModalDataset
from input_utils.yaml_utils import load_yaml


def create_dataloader(option, args, batch_size=64, workers=5):
    """create the dataloader for the given data path.

    Args:
        option (_type_): training, validation, testing dataset
        data_path (_type_): _description_
        workers (_type_): _description_
    """
    # select the index file
    label_ratio = 1
    if option == "train":
        if args.train_mode != "MAE" and args.stage == "pretrains":
            "self-supervised training"
            index_file = args.dataset_config["pretrain_index_file"]
        else:
            """supervised training"""
            index_file = args.dataset_config[args.task]["train_index_file"]
            label_ratio = args.label_ratio
    elif option == "val":
        index_file = args.dataset_config[args.task]["val_index_file"]
    else:
        index_file = args.dataset_config[args.task]["test_index_file"]

    # init the dataset
    triplet_flag = False
    balanced_sample_flag = (
        args.balanced_sample and option == "train" and (args.train_mode == "supervised" or args.stage == "finetune")
    )
    dataset = MultiModalDataset(args, index_file, label_ratio, balanced_sample_flag)
    batch_size = min(batch_size, len(dataset))

    # define the dataloader with weighted sampler for training
    if balanced_sample_flag:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.sample_weights, dataset.epoch_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
        logging.info("=\tUsing class balanced sampler for training")
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(option == "train"), num_workers=workers)

    return dataloader, triplet_flag


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
