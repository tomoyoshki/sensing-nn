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

# DINO utils
from models.DINOModules import DINOWrapper, DINOHead
from models.SimCLRModules import SimCLR


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
    # model config
    classifier_config = args.dataset_config[args.model]
    contrastive_model = None
    if args.contrastive_framework == "SimCLR":
        default_model = SimCLR(args, backbone_model)
        default_model = default_model.to(args.device)
    elif args.contrastive_framework == "DINO":
        default_model = DINOWrapper(
            backbone_model, DINOHead(classifier_config["loc_out_channels"], 1024), args=args.dataset_config
        )
        contrastive_model = DINOWrapper(
            backbone_model, DINOHead(classifier_config["loc_out_channels"], 1024), args=args.dataset_config
        )
        default_model, contrastive_model = default_model.to(args.device), contrastive_model.to(args.device)

        contrastive_model.load_state_dict(default_model.state_dict())
        for p in contrastive_model.parameters():
            p.requires_grad = False
    else:
        raise NotImplementedError

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, default_model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

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
    best_val_acc = 0

    if args.train_mode == "supervised":
        best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_best.pt")
        latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_latest.pt")
    else:
        best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
        latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")

    for epoch in range(classifier_config["lr_scheduler"]["train_epochs"]):
        # set model to train mode
        default_model.train()
        augmenter.train()

        # training loop
        train_loss_list = []

        # regularization configuration
        for i, (time_loc_inputs, _) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            optimizer.zero_grad()
            aug_freq_loc_inputs_1 = augmenter.forward_random(time_loc_inputs)
            aug_freq_loc_inputs_2 = augmenter.forward_random(time_loc_inputs)
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

        # compute
        knn_estimator = compute_knn(args, default_model.backbone, augmenter, train_dataloader)

        # validation and logging
        train_loss = np.mean(train_loss_list)
        val_acc = val_and_logging(
            args,
            epoch,
            tb_writer,
            default_model.backbone,
            augmenter,
            val_dataloader,
            test_dataloader,
            nn.CrossEntropyLoss(),
            train_loss,
            estimator=knn_estimator,
        )

        # Save the latest model
        torch.save(default_model.backbone.state_dict(), latest_weight)

        # Save the best model according to validation result
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(default_model.backbone.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
