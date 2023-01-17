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
        default="Parkland",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="TransformerV3",
        help="The backbone classification model to use.",
    )

    parser.add_argument(
        "-contrastive_framework",
        type=str,
        default="SimCLR",
        help="Contrastive learning framework to use",
    )

    # training and inference mode
    parser.add_argument(
        "-train_mode",
        type=str,
        default="supervised",
        help="The used mode for model training (supervised/original/contrastive/...).",
    )
    parser.add_argument(
        "-inference_mode",
        type=str,
        default="original",
        help="The used mode for model inference (original).",
    )
    parser.add_argument(
        "-stage",
        type=str,
        default="pretrain",
        help="The pretrain/finetune, used for foundation model only.",
    )

    # used for separate training and inference
    parser.add_argument(
        "-miss_modalities",
        type=str,
        default=None,
        help="Specify the unused modalities separated by ,",
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
        default=256,
        help="Specify the batch size for training.",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=3,
        help="Specify which GPU to use.",
    )

    # specify whether to show detailed logs
    parser.add_argument(
        "-verbose",
        type=str,
        default="false",
        help="Whether to show detailed logs.",
    )

    args = parser.parse_args()

    # set option first
    args.option = option

    return args
