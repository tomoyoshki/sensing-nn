from ast import arg
import os
import argparse
from tkinter.messagebox import NO
import numpy as np

from params.output_paths import set_model_weight_file, set_output_paths, set_model_weight_folder
from params.params_util import *
from input_utils.yaml_utils import load_yaml


def parse_base_args(option="train"):
    """
    Parse the args.
    """
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument(
        "-dataset",
        type=str,
        default="RealWorld_HAR",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="DeepSense",
        help="The backbone classification model to use.",
    )

    # training and inference mode
    parser.add_argument(
        "-train_mode",
        type=str,
        default="noisy",
        help="The used mode for model training (original/separate/random/noisy).",
    )
    parser.add_argument(
        "-inference_mode",
        type=str,
        default="noisy",
        help="The used mode for model inference (original/separate/random/noisy).",
    )
    parser.add_argument(
        "-stage",
        type=str,
        default="pretrain_handler",
        help="The train/inference stage for random/noisy modes, pretrain_classifier/pretrain_handler/finetune.",
    )
    parser.add_argument(
        "-elastic_mod",
        type=str,
        default="true",
        help="Whether to enable random modality miss during the pretraining of backbone model.",
    )

    # model, miss generator, tracker and handler configs
    parser.add_argument(
        "-miss_modalities",
        type=str,
        default=None,
        help="Used in inference and train-separated, providing the missing modalities separated by ,",
    )
    parser.add_argument(
        "-miss_detector",
        type=str,
        default="FakeDetector",
        help="The approach used to detect the noise.",
    )
    parser.add_argument(
        "-miss_handler",
        type=str,
        default="FakeHandler",
        help="The approach used to handle the missing modalities.",
    )

    # related to noise generator
    parser.add_argument(
        "-noise_position",
        type=str,
        default="feature",
        help="The noise could be added to time/frequency/feature",
    )
    parser.add_argument(
        "-noise_std_multipler",
        type=float,
        default=5,
        help="The standard deviation of the added noise in the noisy generator",
    )
    parser.add_argument(
        "-noise_mode",
        type=str,
        default="fixed_gaussian",
        help="The mode of noise to be added to the data",
    )

    # weight path
    parser.add_argument(
        "-model_weight",
        type=str,
        default=None,
        help="Specify the model weight path to evaluate.",
    )

    # hardware config
    parser.add_argument(
        "-batch_size",
        type=int,
        default=64,
        help="Specify the batch size for training.",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=None,
        help="Specify which GPU to use.",
    )

    # specify whether to show detailed logs
    parser.add_argument(
        "-verbose",
        type=str,
        default="false",
        help="Whether to show detailed logs.",
    )

    # training configurations
    parser.add_argument(
        "-lr",
        type=float,
        default=None,
        help="Specify the learning rate to try.",
    )

    # evaluation configurations
    parser.add_argument(
        "-eval_detector",
        type=str,
        default="false",
        help="Whether to evaluate the noise detector",
    )
    parser.add_argument(
        "-save_emb",
        type=str,
        default="false",
        help="Whether to save the encoded embeddings.",
    )
    parser.add_argument(
        "-test_noisy_parkland",
        type=str,
        default="false",
        help="Whether to test on the nosiy data of parkland dataset.",
    )
    parser.add_argument(
        "-test_wind_parkland",
        type=str,
        default="false",
        help="Whether to test on the nosiy data of parkland dataset.",
    )

    args = parser.parse_args()

    # set option first
    args.option = option

    return args
