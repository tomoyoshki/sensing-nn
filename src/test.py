from numpy import argsort
import torch.nn as nn

# import models
from models.MissSimulator import MissSimulator
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_given_model, eval_miss_detector


def test(args):
    """The main function for test."""
    # Init data loaders
    test_dataloader, _ = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Init the miss modality simulator
    miss_simulator = MissSimulator(args)
    miss_simulator.to(args.device)
    args.miss_simulator = miss_simulator

    # Load the miss handler
    if miss_simulator.miss_detector.trainable and args.detector_weight is not None:
        miss_simulator.miss_detector = load_model_weight(miss_simulator.miss_detector, args.detector_weight)

    # Load the miss detector
    if miss_simulator.miss_handler.trainable and args.handler_weight is not None:
        miss_simulator.miss_handler = load_model_weight(miss_simulator.miss_handler, args.handler_weight)

    # Init the classifier model
    if args.model == "DeepSense":
        classifier = DeepSense(args, self_attention=False)
    elif args.model == "SADeepSense":
        classifier = DeepSense(args, self_attention=True)
    elif args.model == "Transformer":
        classifier = Transformer(args)
    elif args.model == "TransformerV2":
        classifier = TransformerV2(args)
    elif args.model == "TransformerV3":
        classifier = TransformerV3(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")
    classifier = classifier.to(args.device)
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

    # print resilient handler parameters
    if "ResilientHandler" in args.miss_handler:
        print("----------------------Resilient Handler Parameters----------------------")
        for name, param in miss_simulator.miss_handler.named_parameters():
            print(f"{name} \n", param.data.cpu().numpy())
        print("------------------------------------------------------------------------")

    test_classifier_loss, test_handler_loss, test_f1, test_acc, test_conf_matrix = eval_given_model(
        args, classifier, miss_simulator, test_dataloader, classifier_loss_func
    )
    print(f"Test classifier loss: {test_classifier_loss: .5f}, test handler loss: {test_handler_loss: .5f}")
    print(f"Test acc: {test_acc: .5f}, test f1: {test_f1: .5f}")
    print(f"Test confusion matrix:\n {test_conf_matrix}")
    if args.eval_detector:
        eval_miss_detector(args, "Test", miss_simulator, eval_roc=True)


def main_test():
    """The main function of training"""
    args = parse_test_params()

    test(args)


if __name__ == "__main__":
    main_test()
