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


def mae_train(
    args,
    classifier,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_func,
    tb_writer,
    num_batches,
):
    """
    The mae training function for the backbone network,
    used in train of mae mode of foundation models.
    """
    # model config
    classifier_config = args.dataset_config[args.model]

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, classifier.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    # Fix the patch embedding layer, from MOCOv3
    for name, param in classifier.named_parameters():
        if "patch_embed" in name:
            param.requires_grad = False

    # Print the trainable parameters
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                logging.info(name)

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_loss = np.inf
    best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")
    val_epochs = 5 if args.dataset == "Parkland" else 3
    for epoch in range(classifier_config["lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        classifier.train()
        args.epoch = epoch

        # training loop
        train_loss_list = []

        # regularization configuration
        for i, (time_loc_inputs, labels, index) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)

            # forward pass
            decoded_x, padded_x, masks = classifier(aug_freq_loc_inputs, False)
            loss = loss_func(aug_freq_loc_inputs, decoded_x, masks)

            # back propagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss_list.append(loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train loss", loss.item(), epoch * num_batches + i)

        train_loss = np.mean(train_loss_list)
        logging.info(f"train loss: {train_loss}")
        # validation and logging
        if epoch % val_epochs == 0:
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = val_and_logging(
                args,
                epoch,
                tb_writer,
                classifier,
                augmenter,
                val_dataloader,
                test_dataloader,
                loss_func,
                train_loss,
            )

            # Save the latest model
            torch.save(classifier.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(classifier.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
