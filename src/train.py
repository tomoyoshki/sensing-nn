import warnings

warnings.simplefilter("ignore", UserWarning)

import os
import time
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# from test import eval_given_model
from train_utils.eval_functions import eval_given_model

# import models
from data_augmenter.Augmenter import Augmenter
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3
from models.TransformerV4 import TransformerV4

# train utils
from train_utils.supervised_train import supervised_train_classifier
from train_utils.self_supervised_train import self_supervised_train_classifier
from train_utils.finetune import finetune

# loss functions
from models.loss import DINOLoss

# utils
from torch.utils.tensorboard import SummaryWriter
from params.train_params import parse_train_params
from input_utils.multi_modal_dataloader import create_dataloader, preprocess_triplet_batch


def init_model(args):
    if args.model == "DeepSense":
        classifier = DeepSense(args, self_attention=False)
    elif args.model == "Transformer":
        classifier = Transformer(args)
    elif args.model == "TransformerV2":
        classifier = TransformerV2(args)
    elif args.model == "TransformerV3":
        classifier = TransformerV3(args)
    elif args.model == "TransformerV4":
        classifier = TransformerV4(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")
    return classifier


def train(args):
    """The specific function for training."""
    # Init data loaders
    train_dataloader, triplet_flag = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader, _ = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader, _ = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)
    num_batches = len(train_dataloader)

    logging.info(f"{'='*30}Dataloaders loaded{'='*30}")
    logging.info(f"=\tTrain: {len(train_dataloader)}")
    logging.info(f"=\tVal: {len(val_dataloader)}")
    logging.info(f"=\tTest: {len(test_dataloader)}")
    logging.info(f"{'='*70}")

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_model(args)
    classifier = classifier.to(args.device)

    args.classifier = classifier
    logging.info(f"=\tClassifier model loaded")

    # Init the Tensorboard summary writer
    tb_writer = SummaryWriter(args.tensorboard_log)
    logging.info(f"=\tTensorboard loaded")

    # define the loss function
    if args.multi_class:
        classifier_loss_func = nn.BCELoss()
    else:
        if args.train_mode == "supervised":
            classifier_loss_func = nn.CrossEntropyLoss()
        else:
            # TODO: Setup argument in data yaml file
            classifier_loss_func = DINOLoss().to(args.device)
    logging.info("=\tLoss function defined")

    if args.train_mode == "supervised":
        supervised_train_classifier(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            tb_writer,
            num_batches,
        )
    elif args.train_mode in {"contrastive"}:
        self_supervised_train_classifier(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            tb_writer,
            num_batches,
        )
    else:
        if args.stage == "pretrain_classifier":
            pass
        elif args.stage == "finetune":
            finetune(
                args,
                classifier,
                augmenter,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                tb_writer,
                num_batches,
                triplet_flag,
            )
        else:
            raise Exception(f"Invalid stage provided: {args.stage}")


def main_train():
    """The main function of training"""
    args = parse_train_params()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()]
    )
    train(args)


if __name__ == "__main__":
    main_train()
