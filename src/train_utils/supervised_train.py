import os
import torch
import logging
import torch.optim as optim
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync

# input regularization utils
from input_utils.mixup_utils import Mixup




def supervised_train_classifier(
    args,
    classifier,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    classifier_loss_func,
    tb_writer,
    num_batches,
):
    """
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    # model config
    classifier_config = args.dataset_config[args.model]

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, classifier.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                logging.info(name)

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_acc = 0
    
    # mixup functions
    mixup_func = Mixup(**classifier_config["mixup_args"])

    if args.train_mode == "supervised":
        best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_best.pt")
        latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_latest.pt")
    else:
        best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
        latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")

    for epoch in range(classifier_config["lr_scheduler"]["train_epochs"]):
        # set model to train mode
        classifier.train()
        # augmenter.train()
        args.epoch = epoch

        # training loop
        train_loss_list = []
        
        # regularization configuration
        for i, (data, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # mixup and cutmix augmentation, does nothing if both has alpha 0
            data, labels = mixup_func(data, labels, args.dataset_config)
            # send data label to device (data is sent in the model)
            labels = labels.to(args.device)
            logits = classifier(data, augmenter)
            loss = classifier_loss_func(logits, labels)

            # back propagation
            optimizer.zero_grad()
            loss.backward()

            # clip gradient and update
            torch.nn.utils.clip_grad_norm(classifier.parameters(), classifier_config["optimizer"]["clip_grad"])
            optimizer.step()
            train_loss_list.append(loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train loss", loss.item(), epoch * num_batches + i)

        # validation and logging
        train_loss = np.mean(train_loss_list)
        val_acc = val_and_logging(
            args,
            epoch,
            tb_writer,
            classifier,
            augmenter,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            train_loss,
        )

        # Save the latest model
        torch.save(classifier.state_dict(), latest_weight)

        # Save the best model according to validation result
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
