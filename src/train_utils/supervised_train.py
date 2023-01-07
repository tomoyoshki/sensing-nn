import os
import torch
import logging
import torch.optim as optim
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging

# utils
from general_utils.time_utils import time_sync

# input regularization utils
from input_utils.regularization_utils import rand_bbox
from input_utils.mixup_utils import Mixup
# from timm.data.mixup import Mixup




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

    # Init the optimizer
    lr = args.lr if args.lr is not None else classifier_config["start_lr"]
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                logging.info(name)

    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=classifier_config["scheduler_step"],
        gamma=classifier_config["scheduler_gamma"],
    )

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_acc = 0
    
    mixup_func = Mixup(**classifier_config["mixup_args"])
    if args.train_mode == "supervised":
        best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_best.pt")
        latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_latest.pt")
    else:
        best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
        latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")

    for epoch in range(classifier_config["train_epochs"]):
        # set model to train mode
        classifier.train()
        # augmenter.train()
        args.epoch = epoch

        # training loop
        train_loss_list = []
        
        # regularization configuration
        cutmix_beta = classifier_config["cutmix_beta"]
        for i, (data, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # send data label to device (data is sent in the model)
            data, labels = mixup_func(data, labels, args.dataset_config)
            labels = labels.to(args.device)
            # r = np.random.rand(1)
            # if classifier_config["cutmix_regularization"] and cutmix_beta > 0 and r < classifier_config["cutmix_prob"]:
            #     # generate mixed sample
            #     lam = np.random.beta(cutmix_beta, cutmix_beta)
            #     rand_index = None
            #     for loc in args.dataset_config["location_names"]:
            #         for mod in args.dataset_config["modality_names"]:
            #             if rand_index is None:
            #                 rand_index = torch.randperm(data[loc][mod].size()[0])
            #             bbx1, bby1, bbx2, bby2 = rand_bbox(data[loc][mod].size(), lam)
            #             data[loc][mod][:, :, bbx1:bbx2, bby1:bby2] = data[loc][mod][rand_index, :, bbx1:bbx2, bby1:bby2]

            #     # adjust lambda to exactly match pixel ratio
            #     # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
                
            #     labels_a = labels
            #     labels_b = labels[rand_index.cuda()]

            #     # compute output
            #     logits = classifier(data, augmenter)
            #     loss = classifier_loss_func(logits, labels_a) * lam + classifier_loss_func(logits, labels_b) * (1. - lam)
            # else:
            #     # compute output
            logits = classifier(data, augmenter)
            loss = classifier_loss_func(logits, labels)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
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
        scheduler.step()

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
