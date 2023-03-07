import torch
import numpy as np


def calc_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx):
    """Eval the contrastive loss for one batch."""
    if args.learn_framework == "CMC":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(aug_freq_loc_inputs)
        loss = loss_func(feature1, feature2, idx)
    elif args.learn_framework == "Cosmo":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        rand_fused_features = default_model(aug_freq_loc_inputs)
        loss = loss_func(rand_fused_features)
    else:
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(aug_freq_loc_inputs_1, aug_freq_loc_inputs_2)
        loss = loss_func(feature1, feature2, idx)

    return loss


def calc_predictive_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    """Eval the predictive loss for one batch."""
    if args.learn_framework == "MTSS":
        aug_freq_loc_inputs, pretrain_labels = augmenter.forward("random", time_loc_inputs, return_aug_id=True)
        pretrain_labels = torch.nn.functional.one_hot(pretrain_labels, num_classes=default_model.num_classes).float()
        pretrain_predictions = default_model(aug_freq_loc_inputs)
    elif args.learn_framework in {"ModPred", "ModPredFusion"}:
        aug_freq_loc_inputs, pretrain_labels = augmenter.forward("random", time_loc_inputs, return_aug_mods=True)
        pretrain_predictions = default_model(aug_freq_loc_inputs)
    else:
        raise NotImplementedError(f"Predictive framwork {args.learn_framework} yet implemented")

    loss = loss_func(pretrain_predictions, pretrain_labels)

    return loss

def calc_generative_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    if args.learn_framework == "MAE":
        aug_freq_loc_inputs = augmenter.forward("random", time_loc_inputs)
        decoded_x, padded_x, masks = default_model(aug_freq_loc_inputs)
        loss = loss_func(padded_x, decoded_x, masks)
    else:
        raise NotImplementedError(f"Generative framwork {args.learn_framework} yet implemented")
    return loss
