import os
import time
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from test import eval_given_model

# import models
from models.MissSimulator import MissSimulator
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3

# train utils
from train_utils.pretrain_classifier import pretrain_classifier
from train_utils.pretrain_handler import pretrain_handler, pretrain_handler_gan
from train_utils.pretrain_detector import pretrain_density_detector, pretrain_detector
from train_utils.finetune import finetune

# utils
from torch.utils.tensorboard import SummaryWriter
from params.train_params import parse_train_params
from input_utils.multi_modal_dataloader import create_dataloader, preprocess_triplet_batch


def train(args):
    """The specific function for training."""
    # Init data loaders
    train_dataloader, triplet_flag = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader, _ = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader, _ = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)
    num_batches = len(train_dataloader)

    # Init the miss modality simulator
    miss_simulator = MissSimulator(args)
    miss_simulator.to(args.device)
    args.miss_simulator = miss_simulator

    # Init the classifier model
    if args.model == "DeepSense":
        classifier = DeepSense(args, self_attention=False)
    elif args.model == "SADeepSense":
        classifier = DeepSense(args, self_attention=True)
    elif args.model == "Transformer":
        classifier = Transformer(args)
    elif args.model == "TransformerV2":
        classifier = TransformerV2(args)
    elif args.model == "TransformerV3":
        classifier = TransformerV3(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")
    classifier = classifier.to(args.device)
    args.classifier = classifier

    # Init the Tensorboard summary writer
    tb_writer = SummaryWriter(args.tensorboard_log)

    # define the loss function
    if args.multi_class:
        classifier_loss_func = nn.BCELoss()
    else:
        classifier_loss_func = nn.CrossEntropyLoss()

    if args.stage == "pretrain_classifier":
        pretrain_classifier(
            args,
            classifier,
            miss_simulator,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            tb_writer,
            num_batches,
        )
    elif args.stage == "pretrain_handler":
        if args.miss_handler == "AdAutoencoder":
            pretrain_handler_gan(
                args,
                classifier,
                miss_simulator,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                tb_writer,
                num_batches,
                triplet_flag,
            )
        else:
            pretrain_handler(
                args,
                classifier,
                miss_simulator,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                tb_writer,
                num_batches,
                triplet_flag,
            )
    elif args.stage == "pretrain_detector":
        if args.miss_detector == "DensityDetector":
            pretrain_density_detector(
                args,
                classifier,
                miss_simulator,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                tb_writer,
                num_batches,
                triplet_flag,
            )
        else:
            pretrain_detector(
                args,
                classifier,
                miss_simulator,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                tb_writer,
                num_batches,
                triplet_flag,
            )
    elif args.stage == "finetune":
        finetune(
            args,
            classifier,
            miss_simulator,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            tb_writer,
            num_batches,
            triplet_flag,
        )
    else:
        pass


def main_train():
    """The main function of training"""
    args = parse_train_params()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()]
    )
    train(args)


if __name__ == "__main__":
    main_train()
