import os
import torch
import logging
import torch.optim as optim
import numpy as np
import time

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync



def single_epoch_train(
    args,
    classifier,
    augmenter,
    optimizer,
    epoch,
    train_dataloader,
    loss_func,
    tb_writer,
    num_batches,
    train_loss_list
):
    """
    Single epoch training function for the backbone network
    """

     # regularization configuration
    for i, (time_loc_inputs, labels, _) in tqdm(enumerate(train_dataloader), total=num_batches):
        # move to target device, FFT, and augmentations
        aug_freq_loc_inputs, labels = augmenter.forward("fixed", time_loc_inputs, labels)
        # forward pass
        logits = classifier(aug_freq_loc_inputs)
        loss = loss_func(logits, labels)

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

    
def single_epoch_joint_quantization(
    args,
    classifier,
    augmenter,
    optimizer,
    bitwidth_options,
    joint_quantization_batch_size,
    epoch,
    train_dataloader,
    loss_func,
    tb_writer,
    num_batches,
    train_loss_list
):
     # regularization configuration
    quantized_conv_class = classifier.Conv # QuanConv 
    for i, (time_loc_inputs, labels, _) in tqdm(enumerate(train_dataloader), total=num_batches):
        # move to target device, FFT, and augmentations
        aug_freq_loc_inputs, labels = augmenter.forward("fixed", time_loc_inputs, labels)
        # forward pass
        accumulated_loss = 0.0
        # joint quantization
        for j in range(joint_quantization_batch_size):
            quantized_conv_class.set_random_bitwidth_all()
            # Accumulate Loss here

            logits = classifier(aug_freq_loc_inputs)
            current_loss = loss_func(logits, labels)

            accumulated_loss += current_loss

        avg_loss = accumulated_loss / joint_quantization_batch_size

        # back propagation
        optimizer.zero_grad()
        avg_loss.backward()

        # clip gradient and update
        # torch.nn.utils.clip_grad_norm(classifier.parameters(), classifier_config["optimizer"]["clip_grad"])
        optimizer.step()
        train_loss_list.append(avg_loss.item())

        # Write train log
        if i % 200 == 0:
            tb_writer.add_scalar("Train/Train loss", avg_loss.item(), epoch * num_batches + i)
    

def supervised_train(
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
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    # model config
    classifier_config = args.dataset_config[args.model]
    quantization_config = args.dataset_config["quantization"]

    if quantization_config["enable"]:
    # Setup Conv quantization Functions
        for layer_name, layer in classifier.Conv.layer_registry.items():
            layer.setup_quantize_funcs(
                weight_quantization=quantization_config["weight_quantization"],
                activation_quantization=quantization_config["activation_quantization"],
                bitwidth_opts=quantization_config["bitwidth_options"]
            ) # Setup queantization function, in case of PACT it will also setup switchable alpha

            if quantization_config["sat_weight_normalization"]:
                layer.sat_weight_scaling = True
            else:
                layer.sat_weight_scaling = False

        # Setup CustomBatchNorm Functions
        for layer_name, layer in classifier.bn.layer_registry.items():
            if quantization_config["bn_type"] == "float":
                layer.set_to_float()
            elif quantization_config["bn_type"] == "transitional":
                layer.set_to_transitional(quantization_config["bitwidth_options"])
            elif quantization_config["bn_type"] == "switch":
                layer.set_to_switchable(quantization_config["bitwidth_options"])
            else:
                raise ValueError("\
                    quantization_config['bn_type'] is not set as needed in the dataset config file. Valid options are float, transitional, switch.\
                ")
        #  self.switchable = False
        # self.transitional = False
        # self.floating_point = True
        if quantization_config["bn_type"] == "float":
            verify_float = True
            for layer_name, layer in classifier.bn.layer_registry.items():
                # breakpoint()
                if not layer.switchable and not layer.transitional and layer.floating_point:
                    pass
                else:
                    raise ValueError(f"BatchNorm Layer {layer_name} is not set to float mode")
            logging.info("All BatchNorm layers are set to float mode")
                

    elif not quantization_config["enable"]:
        for layer_name, layer in classifier.Conv.layer_registry.items():
            layer.float_mode = True

        for layer_name, layer in classifier.bn.layer_registry.items():
            layer.set_to_float()

     
    # breakpoint()
    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, classifier.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    # Print the trainable parameters
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                logging.info(name)

    # Training loop
    logging.info("---------------------------Start Training Classifier-------------------------------")
    start = time_sync()

    if "regression" in args.task:
        best_mae = np.inf
    else:
        best_val_acc = 0

    best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.task}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.task}_latest.pt")
    val_epochs = 5 if args.dataset == "Parkland" else 1
    for epoch in range(classifier_config["lr_scheduler"]["train_epochs"]):
        begin_time = time.time()
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        classifier.train()
        args.epoch = epoch

        # training loop
        train_loss_list = []

        if not quantization_config["enable"]:
            single_epoch_train(
                args,
                classifier,
                augmenter,
                optimizer,
                epoch,
                train_dataloader,
                loss_func,
                tb_writer,
                num_batches,
                train_loss_list
            )
        elif quantization_config["enable"]:
            if quantization_config["joint_quantization"]:
                single_epoch_joint_quantization(
                    args,
                    classifier,
                    augmenter,
                    optimizer,
                    bitwidth_options=quantization_config["bitwidth_options"],
                    joint_quantization_batch_size=quantization_config["joint_quantization_batch_size"],
                    epoch=epoch,
                    train_dataloader=train_dataloader,
                    loss_func=loss_func,
                    tb_writer=tb_writer,
                    num_batches=num_batches,
                    train_loss_list=train_loss_list
                )


        # regularization configuration
        # for i, (time_loc_inputs, labels, _) in tqdm(enumerate(train_dataloader), total=num_batches):
        #     # move to target device, FFT, and augmentations
        #     aug_freq_loc_inputs, labels = augmenter.forward("fixed", time_loc_inputs, labels)
        #     # forward pass
        #     logits = classifier(aug_freq_loc_inputs)
        #     loss = loss_func(logits, labels)

        #     # back propagation
        #     optimizer.zero_grad()
        #     loss.backward()

        #     # clip gradient and update
        #     # torch.nn.utils.clip_grad_norm(classifier.parameters(), classifier_config["optimizer"]["clip_grad"])
        #     optimizer.step()
        #     train_loss_list.append(loss.item())

        #     # Write train log
        #     if i % 200 == 0:
        #         tb_writer.add_scalar("Train/Train loss", loss.item(), epoch * num_batches + i)

        end_time = time.time()
        logging.info(f"Epoch {epoch}, Train Loss: {train_loss_list[-1]}")
        logging.info(f"Epoch {epoch} takes {end_time - begin_time:.3f} s")
        # validation and logging
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
                loss_func,
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

    # flush and close the TB writer
    tb_writer.flush()
    tb_writer.close()

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
