import os
import torch
import logging
import numpy as np
from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight


def finetune(
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
    """Fine tune the backbone network with only the class layer."""
    # Load the pretrained feature extractor
    pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
    classifier = load_model_weight(classifier, pretrain_weight, load_class_layer=False)
    learnable_parameters = []
    for name, param in classifier.named_parameters():
        if args.contrastive_framework == "Cosmo":
            if "class_layer" in name or "mod_fusion_layer" in name:
                param.requires_grad = True
                learnable_parameters.append(param)
            else:
                param.requires_grad = False
        else:
            """For SimCLR, MoCo, CMC"""
            if "class_layer" in name:
                param.requires_grad = True
                learnable_parameters.append(param)
            else:
                param.requires_grad = False

    # Init the optimizer
    optimizer = define_optimizer(args, learnable_parameters)
    lr_scheduler = define_lr_scheduler(args, optimizer)

    # Training loop
    logging.info("---------------------------Start Fine Tuning-------------------------------")
    start = time_sync()
    best_val_acc = 0
    best_weight = os.path.join(
        args.weight_folder, f"{args.dataset}_{args.model}_{args.task}_{args.label_ratio}_finetune_best.pt"
    )
    latest_weight = os.path.join(
        args.weight_folder, f"{args.dataset}_{args.model}_{args.task}_{args.label_ratio}_finetune_latest.pt"
    )

    val_epochs = 5 if args.dataset == "Parkland" else 3
    for epoch in range(args.dataset_config[args.contrastive_framework]["finetune_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        classifier.train()

        # training loop
        train_loss_list = []
        for i, (time_loc_inputs, labels, _) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            aug_freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(aug_freq_loc_inputs)
            loss = classifier_loss_func(logits, labels)

            # back propagation
            optimizer.zero_grad()
            loss.backward()

            # clip gradient and update
            # torch.nn.utils.clip_grad_norm(classifier.parameters(), classifier_config["optimizer"]["clip_grad"])
            optimizer.step()
            train_loss_list.append(loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train loss", loss.item(), epoch * num_batches + i)

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
