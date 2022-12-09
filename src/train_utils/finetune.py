import os
import time
import torch
import logging
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from input_utils.multi_modal_dataloader import create_dataloader, preprocess_triplet_batch


def finetune(
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
    """Fine tune the backbone network and the missing-modality handler in an end-to-end manner."""
    # define the miss handler and miss detector
    miss_handler = miss_simulator.miss_handler
    miss_detector = miss_simulator.miss_detector
    miss_generator = miss_simulator.miss_generator

    # handler config
    handler_config = args.dataset_config[args.miss_handler]
    learnable_parameters = []

    # Load the pretrained classifier and handler
    pretrain_classifier_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_best.pt")
    classifier = load_model_weight(classifier, pretrain_classifier_weight)
    for name, param in classifier.named_parameters():
        if "loc_mod_extractors" in name or "loc_mod_feature_extraction_layers" in name:
            param.requires_grad = False
        else:
            learnable_parameters.append(param)

    # Load the handler if it is trainable
    pretrain_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_{args.noise_level}_pretrain_best.pt",
    )
    handler_trainable_flag = miss_handler.trainable
    if handler_trainable_flag:
        miss_handler = load_model_weight(miss_handler, pretrain_handler_weight)

    # Load the detector if it is trainable
    pretrain_detector_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_detector}_{args.noise_level}_pretrain_best.pt",
    )
    detector_trainable_flag = miss_detector.trainable
    if detector_trainable_flag:
        miss_detector = load_model_weight(miss_detector, pretrain_detector_weight)

    # Init the optimizer
    lr = args.lr if args.lr is not None else handler_config["finetune_start_lr"]
    learnable_parameters = learnable_parameters + list(miss_handler.parameters())
    optimizer = optim.Adam(learnable_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=handler_config["finetune_scheduler_step"],
        gamma=handler_config["finetune_scheduler_gamma"],
    )

    # Training loop
    logging.info("---------------------------Start Fine Tuning-------------------------------")
    start = time_sync()
    best_val_acc = 0
    best_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_finetune_best.pt",
    )
    latest_handler_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.miss_handler}_finetune_latest.pt",
    )
    best_classifier_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_on_{args.miss_handler}_finetune_best.pt",
    )
    latest_classifier_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_on_{args.miss_handler}_finetune_latest.pt",
    )

    for epoch in range(handler_config["finetune_epochs"]):

        # set model to train mode
        classifier.train()
        miss_handler.train()
        miss_detector.eval()

        # set the mode for miss generator
        if args.miss_generator == "NoisyGenerator":
            miss_generator.set_noise_mode("random")

        # training loop
        train_classifier_loss_list = []
        train_handler_loss_list = []
        for i, (data, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # preprocess the triplet batch if needed
            data, labels = preprocess_triplet_batch(data, labels) if triplet_flag else (data, labels)

            # send data label to device (data is sent in the model)
            labels = labels.to(args.device)
            logits, handler_loss = classifier(data, miss_simulator)
            if args.miss_handler == "AdAutoencoder":
                g_loss, d_loss = handler_loss
                if epoch % 2 == 0:
                    handler_loss = g_loss
                else:
                    handler_loss = d_loss

            # compute the loss
            classifier_loss = classifier_loss_func(logits, labels)
            loss = (
                handler_config["classifier_loss_weight"] * classifier_loss
                + handler_config["handler_loss_weight"] * handler_loss
            )

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
        torch.save(classifier.state_dict(), latest_classifier_weight)
        if handler_trainable_flag:
            torch.save(miss_handler.state_dict(), latest_handler_weight)

        # Save the best model according to validation result
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), best_classifier_weight)
            if handler_trainable_flag:
                torch.save(miss_handler.state_dict(), best_handler_weight)

        # Update the learning rate scheduler
        scheduler.step()

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
