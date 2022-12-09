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


def pretrain_density_detector(
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
    """The pretraining function for the density-based miss detector."""
    # define the miss handler and miss detector
    miss_handler = miss_simulator.miss_handler
    miss_detector = miss_simulator.miss_detector
    miss_generator = miss_simulator.miss_generator

    # detector config
    detector_config = args.dataset_config[args.miss_detector]

    # Load and freeze the pretrained classifier
    pretrain_classifier_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_best.pt")
    classifier = load_model_weight(classifier, pretrain_classifier_weight)

    # # Load and free the pretrained miss handler
    # pretrain_handler_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.miss_handler}_pretrain_best.pt")
    # handler_trainable_flag = miss_handler.trainable
    # if handler_trainable_flag:
    #     miss_handler = load_model_weight(miss_handler, pretrain_handler_weight)

    # Training loop
    logging.info("---------------------------Start Pretraining Detector-------------------------------")
    start = time_sync()
    best_detector_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_detector}_{args.noise_position}_pretrain_best.pt",
    )
    latest_detector_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_detector}_{args.noise_position}_pretrain_latest.pt",
    )

    for epoch in range(detector_config["pretrain_epochs"]):

        # set model to train mode
        classifier.eval()
        miss_handler.eval()
        miss_detector.train()
        args.epoch = epoch

        # set the noise mode for miss generator
        miss_generator.set_noise_mode("no")

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
            loss = classifier_loss_func(logits, labels)
            train_classifier_loss_list.append(loss.item())
            train_handler_loss_list.append(handler_loss.item())

        # Save the value range
        if args.count_range:
            miss_simulator.save_value_range()

        # for density detector, estimate the parameteres
        miss_detector.estimate_parameters()

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
            tensorboard_logging=False,
            eval_detector=True,
        )

        # Save the latest model
        torch.save(miss_detector.state_dict(), best_detector_weight)

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")


def pretrain_detector(
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
    """The pretraining function for the learnable miss detector."""
    # define the miss handler and miss detector
    miss_handler = miss_simulator.miss_handler
    miss_detector = miss_simulator.miss_detector
    miss_generator = miss_simulator.miss_generator

    # detector config
    detector_config = args.dataset_config[args.miss_detector]

    # Return if the detector is not trainable
    detector_trainable_flag = miss_detector.trainable
    if not detector_trainable_flag:
        logging.info(f"The given miss handler is not trainable: {args.miss_handler}")
        return

    # Load and freeze the pretrained classifier
    pretrain_classifier_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_best.pt")
    classifier = load_model_weight(classifier, pretrain_classifier_weight)
    for param in classifier.parameters():
        param.requires_grad = False

    # Init the optimizer
    if args.lr is not None:
        lr = args.lr
    elif args.noise_position == "feature":
        lr = detector_config["pretrain_start_lr"][f"feature_{args.model}"]
    else:
        lr = detector_config["pretrain_start_lr"][args.noise_position]
    print(f"lr: {lr}")
    optimizer = optim.Adam(miss_detector.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=detector_config["pretrain_scheduler_step"],
        gamma=detector_config["pretrain_scheduler_gamma"],
    )

    # Training loop
    logging.info("---------------------------Start Pretraining Detector-------------------------------")
    start = time_sync()
    min_detector_loss = np.inf
    best_detector_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_detector}_{args.noise_position}_pretrain_best.pt",
    )
    latest_detector_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_detector}_{args.noise_position}_pretrain_latest.pt",
    )

    for epoch in range(detector_config["pretrain_epochs"]):
        # set model to train mode
        classifier.eval()
        miss_handler.eval()
        miss_detector.train()
        args.epoch = epoch

        # set the mode for miss generator
        if args.miss_generator == "NoisyGenerator":
            miss_generator.set_noise_mode("no")

        # training loop
        train_classifier_loss_list = []
        train_detector_loss_list = []
        for i, (data, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # preprocess the triplet batch if needed
            data, labels = preprocess_triplet_batch(data, labels) if triplet_flag else (data, labels)

            # send data label to device (data is sent in the model)
            labels = labels.to(args.device)
            args.labels = labels
            logits, _ = classifier(data, miss_simulator)
            detector_loss = miss_detector.dt_loss

            # compute loss
            classifier_loss = classifier_loss_func(logits, labels)
            loss = (
                detector_config["classifier_loss_weight"] * classifier_loss
                + detector_config["detector_loss_weight"] * detector_loss
            )

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_classifier_loss_list.append(classifier_loss.item())
            train_detector_loss_list.append(detector_loss.item())

            # Write train log
            if i % 200 == 0:
                tb_writer.add_scalar("Train/Train classifier loss", classifier_loss.item(), epoch * num_batches + i)
                tb_writer.add_scalar("Train/Train detector loss", detector_loss.item(), epoch * num_batches + i)

        if args.miss_detector == "VAEPlusDetector" and miss_detector.use_mahalanobis:
            miss_detector.estimate_mahalanobis_params()
            miss_detector.reset_z_cache()

        # Save the latest model
        torch.save(miss_detector.state_dict(), latest_detector_weight)

        # Save the best model
        train_detector_loss = np.mean(train_detector_loss_list)
        logging.info(f"Train detector loss: {train_detector_loss}")
        if train_detector_loss < min_detector_loss:
            min_detector_loss = train_detector_loss
            torch.save(miss_detector.state_dict(), best_detector_weight)

        # validation and logging
        val_and_logging(
            args,
            epoch,
            tb_writer,
            classifier,
            miss_simulator,
            val_dataloader,
            test_dataloader,
            classifier_loss_func,
            (np.mean(train_classifier_loss_list), 0),
            tensorboard_logging=False,
            eval_detector=True,
        )

        # Update the learning rate scheduler
        scheduler.step()

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
