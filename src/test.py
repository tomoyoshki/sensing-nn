import warnings

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# import models
from data_augmenter.Augmenter import Augmenter
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3
from models.TransformerV4 import TransformerV4

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_supervised_model
from train_utils.model_selection import init_model


def test(args):
    """The main function for test."""
    # Init data loaders
    test_dataloader, _ = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_model(args)
    classifier = load_model_weight(classifier, args.classifier_weight)
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

    test_classifier_loss, test_f1, test_acc, test_conf_matrix = eval_supervised_model(
        args, classifier, augmenter, test_dataloader, classifier_loss_func
    )
    print(f"Test classifier loss: {test_classifier_loss: .5f}")
    print(f"Test acc: {test_acc: .5f}, test f1: {test_f1: .5f}")
    print(f"Test confusion matrix:\n {test_conf_matrix}")


def main_test():
    """The main function of training"""
    args = parse_test_params()

    test(args)


if __name__ == "__main__":
    main_test()
