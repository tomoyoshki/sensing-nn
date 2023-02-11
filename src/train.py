import warnings

warnings.simplefilter("ignore", UserWarning)

import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# import models
from data_augmenter.Augmenter import Augmenter

# train utils
from train_utils.eval_functions import eval_supervised_model
from train_utils.supervised_train import supervised_train
from train_utils.contrastive_train import contrastive_pretrain
from train_utils.predictive_train import predictive_pretrain
from train_utils.finetune import finetune

# loss functions
from models.loss import DINOLoss, SimCLRLoss, MoCoLoss, CMCLoss

# utils
from torch.utils.tensorboard import SummaryWriter
from params.train_params import parse_train_params
from input_utils.multi_modal_dataloader import create_dataloader, preprocess_triplet_batch
from input_utils.time_input_utils import count_range
from train_utils.model_selection import init_model


def train(args):
    """The specific function for training."""
    # Init data loaders
    train_dataloader, _ = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
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
    args.classifier = classifier
    logging.info(f"=\tClassifier model loaded")

    # Init the Tensorboard summary writer
    tb_writer = SummaryWriter(args.tensorboard_log)
    logging.info(f"=\tTensorboard loaded")

    # Optional range counting for training data
    if args.count_range:
        logging.info("=\tCounting range for training data")
        count_range(args, train_dataloader)

    # define the loss function
    if args.multi_class:
        loss_func = nn.BCELoss()
    else:
        if args.train_mode == "supervised" or args.stage == "finetune":
            loss_func = nn.CrossEntropyLoss()
        elif args.train_mode == "predictive":
            """Predictive pretraining only."""
            if args.predictive_framework == "MTSS":
                loss_func = nn.CrossEntropyLoss()
        elif args.train_mode == "contrastive":
            """Contrastive pretraining only."""
            if args.contrastive_framework == "DINO":
                loss_func = DINOLoss(args).to(args.device)
            elif args.contrastive_framework == "MoCo":
                loss_func = MoCoLoss(args).to(args.device)
            elif args.contrastive_framework in {"CMC", "Cosmo"}:
                loss_func = CMCLoss(args, len(train_dataloader.dataset)).to(args.device)
            elif args.contrastive_framework in {"SimCLR"}:
                loss_func = SimCLRLoss(args).to(args.device)
            else:
                raise NotImplementedError(f"Loss function for {args.contrastive_framework} yet implemented")
        else:
            raise Exception(f"Invalid train mode provided: {args.train_mode}")
    logging.info("=\tLoss function defined")

    if args.train_mode == "supervised":
        supervised_train(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
        )
    elif args.train_mode == "contrastive" and args.stage == "pretrain":
        contrastive_pretrain(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
        )

    elif args.train_mode == "predictive" and args.stage == "pretrain":
        predictive_pretrain(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
        )
    elif args.stage == "finetune":
        finetune(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            tb_writer,
            num_batches,
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
