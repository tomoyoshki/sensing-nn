import os
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging, eval_contrastive_loss
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler
from train_utils.knn import compute_embedding, compute_knn
from train_utils.model_selection import init_contrastive_framework

# utils
from general_utils.time_utils import time_sync


def contrastive_pretrain(
    args,
    backbone_model,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_func,
    tb_writer,
    num_batches,
):
    """
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    classifier_config = args.dataset_config[args.model]

    # Initialize contrastive model
    default_model = init_contrastive_framework(args, backbone_model)

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, default_model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    # Fix the patch embedding layer, from MOCOv3
    for name, param in default_model.backbone.named_parameters():
        if "patch_embed" in name:
            param.requires_grad = False

    # Print the trainable parameters
    if args.verbose:
        for name, param in default_model.backbone.named_parameters():
            if param.requires_grad:
                logging.info(name)

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_loss = np.inf
    best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")

    for epoch in range(args.dataset_config[args.contrastive_framework]["pretrain_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        default_model.train()

        # training loop
        train_loss_list = []

        # regularization configuration
        for i, (time_loc_inputs, _, idx) in tqdm(enumerate(train_dataloader), total=num_batches):
            # clear the gradients
            optimizer.zero_grad()
            idx = idx.to(args.device)

            # move to target device, FFT, and augmentations
            loss = eval_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx)

            # back propagation
            loss.backward()

            # update
            optimizer.step()
            train_loss_list.append(loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train loss", loss.item(), epoch * num_batches + i)

        if epoch % 10 == 0:
            # compute embedding for the validation dataloader
            embs, imgs, labels_ = compute_embedding(args, default_model.backbone, augmenter, val_dataloader)
            tb_writer.add_embedding(
                embs,
                metadata=labels_,
                label_img=imgs,
                global_step=epoch,
                tag="embeddings",
            )

            # Use KNN classifier for validation
            knn_estimator = compute_knn(args, default_model.backbone, augmenter, train_dataloader)

            # validation and logging
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = val_and_logging(
                args,
                epoch,
                tb_writer,
                default_model,
                augmenter,
                val_dataloader,
                test_dataloader,
                loss_func,
                train_loss,
                estimator=knn_estimator,
            )

            # Save the latest model, only the backbone parameters are saved
            torch.save(default_model.backbone.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(default_model.backbone.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
