import warnings



import torch.nn as nn

# utils
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_supervised_model
from train_utils.model_selection import init_backbone_model

warnings.simplefilter("ignore", UserWarning)


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
    args.classifier_weight = args.classifier_weight.replace("best", "latest")
    classifier = load_model_weight(args, classifier, args.classifier_weight, load_class_layer=True)
    print(f"Weight: {args.classifier_weight}")
    args.classifier = classifier

    # define the loss function
    if "regression" in args.task:
        classifier_loss_func = nn.MSELoss()
    else:
        classifier_loss_func = nn.CrossEntropyLoss()

    # print model layers
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                print(name)
    
    test_classifier_loss, test_metrics = eval_supervised_model(
        args, classifier, augmenter, test_dataloader, classifier_loss_func
    )
    if "regression" in args.task:
        print(f"Test classifier loss: {test_classifier_loss: .5f}, test mse: {test_metrics[0]: .5f}")
        return test_classifier_loss, test_metrics[0]
    else:
        
        print(f"Test classifier loss: {test_classifier_loss: .5f}")
        
        if isinstance(test_metrics[0], tuple):
            for i in range(2):
                print(f"Test acc {i}: {test_metrics[0][i]: .5f}, test f1 {i}: {test_metrics[1][i]: .5f}")
        else:
            print(f"Test acc: {test_metrics[0]: .5f}, test f1: {test_metrics[1]: .5f}")
        if isinstance(test_metrics[2], tuple):
            print(f"Test confusion matrix 1:\n {test_metrics[2][0]}")
            print(f"Test confusion matrix 2:\n {test_metrics[2][1]}")
        else:
            print(f"Test confusion matrix:\n {test_metrics[2]}")

        return test_classifier_loss, test_metrics[0], test_metrics[1]


def main_test():
    """The main function of training"""
    args = parse_test_params()

    test(args)


if __name__ == "__main__":
    main_test()
