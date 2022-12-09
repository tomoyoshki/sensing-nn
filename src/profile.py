from numpy import argsort
import torch.nn as nn
import os
import json
import torch
import time

# import models
from models.MissSimulator import MissSimulator
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer

# utils
from params.base_params import parse_base_args
from params.params_util import set_auto_params
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
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

    # eval the backbone model
    test_classifier_loss, test_handler_loss, test_f1, test_acc, test_conf_matrix = eval_given_model(
        args, classifier, miss_simulator, test_dataloader, classifier_loss_func
    )

    # eval the miss detector
    det_auc, mean_score = eval_miss_detector(args, "Test", miss_simulator)

    return test_acc, test_f1, det_auc, mean_score


def test_given_dataset_model_noise(dataset, model, miss_detector, noise_mode, noise_std_multipler):
    """The main function of training"""
    args = parse_base_args("test")
    torch.cuda.empty_cache()

    # Update the experiment config params
    args.dataset = dataset
    args.model = model
    args.noise_mode = noise_mode
    args.noise_std_multipler = noise_std_multipler
    args.miss_detector = miss_detector

    # set auto params
    args = set_auto_params(args)

    # measure the acc, f1
    test_acc, test_f1, det_auc, mean_score = test(args)
    # print(dataset, model, noise_mode, noise_std_multipler, test_acc, test_f1, det_auc)

    return test_acc, test_f1, det_auc, mean_score


if __name__ == "__main__":
    """dataset --> model --> noise_mode --> noise_std_multipler --> [acc, f1, detector_auc, detector_mean_score]"""
    log_file = "/home/sl29/AutoCuration/result/log/backbone_sensitivity_density.json"

    dataset_list = ["RealWorld_HAR", "WESAD", "Parkland"]
    model_list = ["DeepSense", "ResNet", "Transformer"]
    noise_mode_list = ["fixed_gaussian"]
    detector_list = ["DensityDetector"]
    noise_std_multipler_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8]
    total_runs = 5

    if not os.path.exists(log_file):
        buffer = {}
    else:
        buffer = json.load(open(log_file, "r"))

    start = time.time()
    for dataset in dataset_list:
        if dataset not in buffer:
            buffer[dataset] = {}
        for model in model_list:
            if model not in buffer[dataset]:
                buffer[dataset][model] = {}
            for noise_mode in noise_mode_list:
                if noise_mode not in buffer[dataset][model]:
                    buffer[dataset][model][noise_mode] = {}
                for noise_std_multipler in noise_std_multipler_list:
                    str_std = str(noise_std_multipler)
                    if str_std not in buffer[dataset][model][noise_mode]:
                        buffer[dataset][model][noise_mode][str_std] = {}

                    buffer[dataset][model][noise_mode][str_std]["acc"] = []
                    buffer[dataset][model][noise_mode][str_std]["f1"] = []

                    # run loops
                    for detector in detector_list:
                        print(f"Testing {dataset} {model} {detector} {noise_mode} {noise_std_multipler}.")
                        buffer[dataset][model][noise_mode][str_std][f"{detector}_auc"] = []
                        buffer[dataset][model][noise_mode][str_std][f"{detector}_mean_score"] = []
                        for i in range(total_runs):
                            acc, f1, det_auc, mena_score = test_given_dataset_model_noise(
                                dataset, model, detector, noise_mode, noise_std_multipler
                            )
                            buffer[dataset][model][noise_mode][str_std]["acc"].append(acc)
                            buffer[dataset][model][noise_mode][str_std]["f1"].append(f1)
                            buffer[dataset][model][noise_mode][str_std][f"{detector}_auc"].append(det_auc)
                            buffer[dataset][model][noise_mode][str_std][f"{detector}_mean_score"].append(mena_score)

                    # write to file after each config
                    with open(log_file, "w") as f:
                        f.write(json.dumps(buffer, indent=4))

    end = time.time()
    print("-" * 60)
    print(f"Total time: {end - start: .2f} s")
