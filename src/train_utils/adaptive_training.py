import os
import torch
import logging
import torch.optim as optim
import numpy as np
import time
# import torch
import torch.nn as nn
from functools import reduce
from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync

class KurtosisLossCalc:
    def __init__(self, weight_tensor, kurtosis_target=1.8, k_mode='avg'):
        self.kurtosis_loss = 0
        self.kurtosis = 0
        self.weight_tensor = weight_tensor
        self.k_mode = k_mode
        self.kurtosis_target = kurtosis_target

    def fn_regularization(self):
        return self.kurtosis_calc()

    def kurtosis_calc(self):
        mean_output = torch.mean(self.weight_tensor)
        std_output = torch.std(self.weight_tensor)
        kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
        self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
        self.kurtosis = kurtosis_val

        if self.k_mode == 'avg':
            self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.mean(kurtosis_val)
        elif self.k_mode == 'max':
            self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.max(kurtosis_val)
        elif self.k_mode == 'sum':
            self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.sum(kurtosis_val)


def KurtosisLoss(model,target):
    KurtosisList = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, model.Conv):
            w_kurt_inst = KurtosisLossCalc(m.weight,kurtosis_target=target)
            w_kurt_inst.fn_regularization()
            KurtosisList.append(w_kurt_inst.kurtosis_loss)
    del KurtosisList[0]
    # del KurtosisList[-1]
    w_kurtosis_loss = reduce((lambda a, b: a + b), KurtosisList) / len(KurtosisList)
    w_kurtosis_regularization = w_kurtosis_loss
    return w_kurtosis_regularization


class TeacherStudentLossCalc:
    def __init__(self, classifier, strategy="activation_min"):
        self.classifier = classifier
        self.strategy = strategy
        self.loss = 0.0
        
    def calculate_kl_divergence(self, student_activation, teacher_activation):
        """Calculate KL divergence between student and teacher activations"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        
        # Normalize activations to create probability distributions
        student_prob = torch.nn.functional.softmax(student_activation.view(-1), dim=0) + epsilon
        teacher_prob = torch.nn.functional.softmax(teacher_activation.view(-1), dim=0) + epsilon
        
        # Calculate KL divergence
        kl_div = torch.sum(teacher_prob * torch.log(teacher_prob / student_prob))
        return kl_div

    def calculate_mse(self, student_activation, teacher_activation):
        """Calculate mean square root difference between activations"""
        diff = student_activation - teacher_activation
        return torch.sqrt(torch.mean(diff**2))

    def calculate_loss(self):
        """Calculate loss based on strategy"""
        total_loss = 0.0
        num_layers = 0
        
        # Iterate through all QuanConv layers in classifier
        for name, layer in self.classifier.named_modules():
            if isinstance(layer, self.classifier.Conv):  # QuanConv layers
                if layer.recent_student_activation is not None and layer.best_teacher_activation is not None:
                    if self.strategy == "activation_min":
                        layer_loss = self.calculate_kl_divergence(
                            layer.recent_student_activation,
                            layer.best_teacher_activation
                        )
                    elif self.strategy == "weight_min":
                        layer_loss = self.calculate_mse(
                            layer.recent_student_activation,
                            layer.best_teacher_activation
                        )
                    else:
                        raise ValueError(f"Unknown strategy: {self.strategy}")
                    
                    total_loss += layer_loss
                    num_layers += 1

        # Average loss across all layers
        if num_layers > 0:
            self.loss = total_loss / num_layers
        return self.loss

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
        logits, _ = classifier(aug_freq_loc_inputs)
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
    quantization_config = args.dataset_config["quantization"]

    for i, (time_loc_inputs, labels, _) in tqdm(enumerate(train_dataloader), total=num_batches):
        # # move to target device, FFT, and augmentations
        # if 'coherence' in time_loc_inputs:
        #     coherence = time_loc_inputs['coherence']
        # if 'energy' in time_loc_inputs:
        #     energy = time_loc_inputs['energy']

        # # if coherence is not None and energy is not None:
        # #     if coherence.shape != energy.shape:
        # #         raise ValueError("Coherence and Energy should be of same shape")
            
        # si_weights = get_si_weights(coherence, energy) # Get weights using coherence and energy <- si_weights is a dictionary of weights
        # # for both modalities. si_weights = {'audio': audio_weight, 'seismic': seismic_weight}

        # assert coherence is not None and energy is not None and coherence.shape == energy.shape, "Coherence and Energy should be of same shape"
        # time_loc_inputs = time_loc_inputs['data']

        aug_freq_loc_inputs, labels = augmenter.forward("fixed", time_loc_inputs, labels)
        # forward pass
        accumulated_loss = 0.0
        # joint quantization
        for j in range(joint_quantization_batch_size):
            quantized_conv_class.set_random_bitwidth_all()
            # Accumulate Loss here

            logits_student, logits_teacher = classifier(aug_freq_loc_inputs)
            current_loss = loss_func(logits = logits_student,  targets =labels, teacher_logits = logits_teacher) # Labels are true values
            accumulated_loss += current_loss

        avg_loss = accumulated_loss / joint_quantization_batch_size

        # TODO: Weight Regularization Loss too for neatness
        if quantization_config["enable"] and quantization_config["kurtosis_regularization"]:
            w_kurtosis_regularization = KurtosisLoss(classifier, quantization_config["kurtosis_target"])
            avg_loss += quantization_config["kurtosis_weight"]*w_kurtosis_regularization   

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
                args
            ) # Setup queantization function, in case of PACT it will also setup switchable alpha

        # Setup CustomBatchNorm Functions
        for layer_name, layer in classifier.bn.layer_registry.items():
            layer.setup_quantize_funcs(
                args
            )
           
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

     
    #Move classifier to device
    target_device = args.device
    classifier.to(target_device)
    logging.info(f"=\tClassifier moved to {target_device} after setting up BatchNorm and Conv layers")
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
            if quantization_config["training_type"] == "joint_quantization":
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
            elif quantization_config["training_type"] == "vanilla":
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
            else:
                raise ValueError(f"Unknown training type: {quantization_config['training_type']}")

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
