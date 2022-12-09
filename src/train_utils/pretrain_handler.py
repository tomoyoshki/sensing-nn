import os
import time
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from input_utils.multi_modal_dataloader import create_dataloader, preprocess_triplet_batch


def pretrain_handler(
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
):
    """The pretraining function for the missing modality handler."""
    # define the miss handler and miss detector
    miss_handler = miss_simulator.miss_handler
    miss_detector = miss_simulator.miss_detector
    miss_generator = miss_simulator.miss_generator

    # handler config
    handler_config = args.dataset_config[args.miss_handler]

    # Return if the handler is not trainable
    handler_trainable_flag = miss_handler.trainable
    if not handler_trainable_flag:
        logging.info(f"The given miss handler is not trainable: {args.miss_handler}")
        return

    # Load and freeze the pretrained classifier
    pretrain_classifier_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_pretrain_best.pt",
    )
    classifier = load_model_weight(classifier, pretrain_classifier_weight)
    for name, param in classifier.named_parameters():
        param.requires_grad = False

    # Load the detector if it is trainable
    pretrain_detector_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_detector}_{args.noise_position}_pretrain_best.pt",
    )
    detector_trainable_flag = miss_detector.trainable
    if detector_trainable_flag:
        miss_detector = load_model_weight(miss_detector, pretrain_detector_weight)
        for name, param in miss_detector.named_parameters():
            param.requires_grad = False

    # Init the optimizer
    if args.lr is not None:
        lr = args.lr
    elif args.miss_handler in {"ResilientHandler", "NonlinearResilientHandler"}:
        if args.noise_position == "feature":
            lr = handler_config["pretrain_start_lr"][f"feature_{args.model}"]
        else:
            lr = handler_config["pretrain_start_lr"][args.noise_position]
    else:
        lr = handler_config["pretrain_start_lr"]
    optimizer = optim.Adam(miss_handler.parameters(), lr=lr)
    print(f"lr: {lr}")

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=handler_config["pretrain_scheduler_step"],
        gamma=handler_config["pretrain_scheduler_gamma"],
    )

    # Training loop
    logging.info("---------------------------Start Pretraining Handler-------------------------------")
    start = time_sync()
    best_val_acc = 0
    best_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_{args.noise_position}_pretrain_best.pt",
    )
    latest_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_{args.noise_position}_pretrain_latest.pt",
    )

    for epoch in range(handler_config["pretrain_epochs"]):

        # set handler to train mode
        classifier.eval()
        miss_detector.eval()
        miss_handler.train()
        args.epoch = epoch

        # set RNN layer to train
        if args.model == "DeepSense":
            classifier.recurrent_layer.train()

        # set the mode for miss generator
        if args.miss_generator == "NoisyGenerator":
            miss_generator.set_noise_mode("random_gaussian")

        # training loop
        train_classifier_loss_list = []
        train_handler_loss_list = []
        for i, (data, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # preprocess the triplet batch if needed
            data, labels = preprocess_triplet_batch(data, labels) if triplet_flag else (data, labels)

            # send data label to device (data is sent in the model)
            labels = labels.to(args.device)
            args.labels = labels
            logits, handler_loss = classifier(data, miss_simulator)

            # compute loss
            classifier_loss = classifier_loss_func(logits, labels)
            loss = (
                handler_config["classifier_loss_weight"] * classifier_loss
                + handler_config["handler_loss_weight"] * handler_loss
            )

            # for name, param in miss_handler.named_parameters():
            #     print(name, param.data.cpu().numpy())

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_classifier_loss_list.append(classifier_loss.item())
            train_handler_loss_list.append(handler_loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train classifier loss", classifier_loss.item(), epoch * num_batches + i)
                tb_writer.add_scalar("Train/Train handler loss", handler_loss.item(), epoch * num_batches + i)

        # validation and logging
        train_classifier_loss = np.mean(train_classifier_loss_list)
        train_handler_loss = np.mean(train_handler_loss_list)
        val_acc = val_and_logging(
            args,
            epoch,
            tb_writer,
            classifier,
            miss_simulator,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            (train_classifier_loss, train_handler_loss),
        )

        # Save the latest model
        torch.save(miss_handler.state_dict(), latest_handler_weight)

        # Save the best model according to validation result
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(miss_handler.state_dict(), best_handler_weight)

        # Update the learning rate scheduler
        scheduler.step()

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")


def pretrain_handler_gan(
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
):
    """The pretraining function for the missing modality handler."""
    # define the miss handler and miss detector
    miss_handler = miss_simulator.miss_handler
    miss_detector = miss_simulator.miss_detector
    miss_generator = miss_simulator.miss_generator

    # handler config
    handler_config = args.dataset_config[args.miss_handler]

    # Return if the handler is not trainable
    handler_trainable_flag = miss_handler.trainable
    if not handler_trainable_flag:
        logging.info(f"The given miss handler is not trainable: {args.miss_handler}")
        return

    # Load and freeze the pretrained classifier
    pretrain_classifier_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_pretrain_best.pt",
    )
    classifier = load_model_weight(classifier, pretrain_classifier_weight)
    for param in classifier.parameters():
        param.requires_grad = False

    # Init the optimizer
    lr = args.lr if args.lr is not None else handler_config["pretrain_start_lr"]
    optimizer_G = optim.Adam(miss_handler.generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(miss_handler.discriminator.parameters(), lr=lr)

    scheduler_G = torch.optim.lr_scheduler.StepLR(
        optimizer_G,
        step_size=handler_config["pretrain_scheduler_step"],
        gamma=handler_config["pretrain_scheduler_gamma"],
    )

    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D,
        step_size=handler_config["pretrain_scheduler_step"],
        gamma=handler_config["pretrain_scheduler_gamma"],
    )

    # Training loop
    logging.info("---------------------------Start Pretraining Handler-------------------------------")
    start = time_sync()
    best_val_acc = 0
    best_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_{args.noise_position}_pretrain_best.pt",
    )
    latest_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_{args.noise_position}_pretrain_latest.pt",
    )

    for epoch in range(handler_config["pretrain_epochs"]):

        # set model to train mode
        classifier.eval()
        miss_detector.eval()
        miss_handler.train()
        args.epoch = epoch

        # set RNN layer to train
        if args.model == "DeepSense":
            classifier.recurrent_layer.train()

        # set the mode for miss generator
        if args.miss_generator == "NoisyGenerator":
            miss_generator.set_noise_mode("random_gaussian")

        # training loop
        train_classifier_loss_list = []
        train_handler_loss_list = []
        for i, (data, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # preprocess the triplet batch if needed
            data, labels = preprocess_triplet_batch(data, labels) if triplet_flag else (data, labels)

            # send data label to device (data is sent in the model)
            labels = labels.to(args.device)
            args.labels = labels
            logits, handler_loss = classifier(data, miss_simulator)
            g_loss, d_loss = handler_loss

            # compute loss
            classifier_loss = classifier_loss_func(logits, labels)

            # ----------------
            # Train Generator
            # ----------------
            loss_G = (
                handler_config["classifier_loss_weight"] * classifier_loss
                + handler_config["handler_loss_weight"] * g_loss
            )

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # --------------------
            # Train Discriminator
            # --------------------
            logits, handler_loss = classifier(data, miss_simulator)
            g_loss, d_loss = handler_loss
            classifier_loss = classifier_loss_func(logits, labels)

            loss_D = handler_config["handler_loss_weight"] * d_loss

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            train_classifier_loss_list.append(classifier_loss.item())
            train_handler_loss_list.append(g_loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train classifier loss", classifier_loss.item(), epoch * num_batches + i)
                tb_writer.add_scalar("Train/Train handler loss", g_loss.item(), epoch * num_batches + i)

        # validation and logging
        train_classifier_loss = np.mean(train_classifier_loss_list)
        train_handler_loss = np.mean(train_handler_loss_list)
        val_acc = val_and_logging(
            args,
            epoch,
            tb_writer,
            classifier,
            miss_simulator,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            (train_classifier_loss, train_handler_loss),
        )

        # Save the latest model
        torch.save(miss_handler.state_dict(), latest_handler_weight)

        # Save the best model according to validation result
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(miss_handler.state_dict(), best_handler_weight)

        # Update the learning rate scheduler
        scheduler_G.step()
        scheduler_D.step()

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
