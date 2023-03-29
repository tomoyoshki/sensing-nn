import os
import warnings

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_supervised_model
from train_utils.model_selection import init_backbone_model


def test(args):
    """The main function for test."""
    # Init data loaders
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_backbone_model(args)
    classifier = load_model_weight(classifier, args.classifier_weight, load_class_layer=True)
    args.classifier = classifier

    # define the loss function
    if args.multi_class:
        classifier_loss_func = nn.BCELoss()
    else:
        classifier_loss_func = nn.CrossEntropyLoss()

    # print model layers
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                print(name)

    test_classifier_loss, test_acc, test_f1, test_conf_matrix = eval_supervised_model(
        args, classifier, augmenter, test_dataloader, classifier_loss_func
    )
    print(f"Test classifier loss: {test_classifier_loss: .5f}")
    print(f"Test acc: {test_acc: .5f}, test f1: {test_f1: .5f}")
    print(f"Test confusion matrix:\n {test_conf_matrix}")

    return test_classifier_loss, test_acc, test_f1


def main_test():
    """The main function of training"""
    args = parse_test_params()

    test(args)


if __name__ == "__main__":
    main_test()
