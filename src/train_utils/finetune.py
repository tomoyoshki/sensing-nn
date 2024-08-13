import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import time

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight, set_learnable_params_finetune
from params.output_paths import set_finetune_weights


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
    if args.train_mode == "supervised":
        pretrain_weight = os.path.join(
            args.weight_folder, f"{args.dataset}_{args.model}_{args.task}_best.pt"
        )
    else:
        pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
    classifier = load_model_weight(args, classifier, pretrain_weight, load_class_layer=False)
    learnable_parameters = set_learnable_params_finetune(args, classifier)

    # Init the optimizer, scheduler, and weight files
    optimizer = define_optimizer(args, learnable_parameters)
    lr_scheduler = define_lr_scheduler(args, optimizer)
    best_weight, latest_weight = set_finetune_weights(args)

    # Training loop
    logging.info("---------------------------Start Fine Tuning-------------------------------")
    start = time_sync()
    if "regression" in args.task:
        best_mae = np.inf
    else:
        best_val_acc = 0

    val_epochs = 5 if args.dataset == "Parkland" else 3
    if args.train_mode == "supervised":
        epochs = args.dataset_config[args.model]["lr_scheduler"]["train_epochs"]
    else:
        epochs = args.dataset_config[args.learn_framework]["finetune_lr_scheduler"]["train_epochs"]
    
    time_taken = []
    for epoch in range(epochs):
        epoch_beign_time = time.time()
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        classifier.train()

        # training loop
        train_loss_list = []
        
        # choose one random sample idx:
        for i, (time_loc_inputs, labels, detection_labels, _) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            if "dual" in args.finetune_tag:
                labels = torch.cat((labels, detection_labels), dim=1)

            aug_freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(aug_freq_loc_inputs)

            # if args.multi_class or "multiclass" in args.finetune_tag:
                # labels = labels.reshape(labels.shape[0], -1)
                # detection_labels = labels[:, -1].reshape(labels.shape[0], -1).long()
                # if labels.shape
                # labels = torch.nn.functional.one_hot(labels[:, 0], num_classes=args.num_class).float()
                # if "dual" in args.finetune_tag:
                    # labels = torch.cat((labels, detection_labels), dim=1)

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
        epoch_end_time = time.time()
        
        time_taken.append(epoch_end_time - epoch_beign_time)
        
        logging.info(f"Epoch {epoch} took {epoch_end_time - epoch_beign_time} s")
        if epoch % val_epochs == 0:
            train_loss = np.mean(train_loss_list)
            val_metric, val_loss = val_and_logging(
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
            if "regression" in args.task:
                if val_metric < best_mae:
                    best_mae = val_metric
                    torch.save(classifier.state_dict(), best_weight)
            else:
                if val_metric > best_val_acc:
                    best_val_acc = val_metric
                    torch.save(classifier.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    logging.info(f"Average time taken: {np.mean(time_taken)}")
    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
