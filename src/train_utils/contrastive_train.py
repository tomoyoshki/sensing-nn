import os
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler
from train_utils.knn import compute_embedding, compute_knn

# utils
from general_utils.time_utils import time_sync

# Contrastive Learning utils
from models.DINOModules import DINO
from models.SimCLRModules import SimCLR
from models.MoCoModule import MoCoWrapper


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
        for i, (time_loc_inputs, _) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            optimizer.zero_grad()
            aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
            aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
            feature1, feature2 = default_model(aug_freq_loc_inputs_1, aug_freq_loc_inputs_2)

            # forward pass
            loss = loss_func(feature1, feature2)

            # back propagation
            loss.backward()

            # clip gradient and update
            # torch.nn.utils.clip_grad_norm(
            #     default_model.backbone.parameters(), classifier_config["optimizer"]["clip_grad"]
            # )
            optimizer.step()

            if args.contrastive_framework == "DINO":
                with torch.no_grad():
                    for backbone_ps, teacher_ps in zip(
                        default_model.backbone.parameters(), default_model.teacher.parameters()
                    ):
                        teacher_ps.data.mul_(default_model.config["momentum_teacher"])
                        teacher_ps.data.add_((1 - default_model.config["momentum_teacher"]) * backbone_ps.detach().data)

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

            # Save the latest model
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


def init_contrastive_framework(args, backbone_model):
    # model config
    if args.contrastive_framework == "SimCLR":
        default_model = SimCLR(args, backbone_model)
    elif args.contrastive_framework == "DINO":
        default_model = DINO(args, backbone_model)
    elif args.contrastive_framework == "MoCo":
        default_model = MoCoWrapper(args, backbone_model)
    else:
        raise NotImplementedError
    default_model = default_model.to(args.device)
    return default_model
